# Scattermind distributes computation of machine learning models.
# Copyright (C) 2024 Josua Krause
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Executor that computes the execution graph."""
import time
import traceback
from collections.abc import Callable

from scattermind.system.base import ExecutorId, Module, TaskId
from scattermind.system.graph.node import Node
from scattermind.system.logger.context import (
    add_context,
    ContextInfo,
    get_ctx,
    get_preexc_ctx,
)
from scattermind.system.logger.error import ErrorCode, ErrorInfo
from scattermind.system.logger.log import EventStream
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import ComputeState, NoTasksToCompute
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


class Executor:
    """A wrapper to handle executor control methods more easily.
    This is used to represent (other) executors for operations."""
    def __init__(
            self, emng: 'ExecutorManager', executor_id: ExecutorId) -> None:
        """
        Create an executor object.

        Args:
            emng (ExecutorManager): The executor manager.
            executor_id (ExecutorId): The executor id.
        """
        self._emng = emng
        self._executor_id = executor_id

    def get_id(self) -> ExecutorId:
        """
        Get the executor id.

        Returns:
            ExecutorId: The executor id.
        """
        return self._executor_id

    def is_active(self) -> bool:
        """
        Whether the exxecutor is currently active.

        Returns:
            bool: True if the executor is active.
        """
        return self._emng.is_active(self._executor_id)

    def release(self) -> None:
        """
        Releases the executor and removes it from the pool whenever possible.
        """
        return self._emng.release_executor(self._executor_id)


class ExecutorManager(Module):
    """
    An executor manager handles starting and stopping executors, running the
    execution loop, and various housekeeping operations (such as releasing
    unresponsive executors). In a distributed setting each executor runs an
    executor manager so redundant operations (e.g., housekeeping) must be
    performed in a safe, asynchronous, and idempotent way.
    """
    def __init__(self, own_id: ExecutorId, batch_size: int) -> None:
        """
        Creates an executor manager for the given executor id.

        Args:
            own_id (ExecutorId): The executor id.
            batch_size (int): The batch size for processing tasks of the
                given executor.
        """
        self._node: Node | None = None
        self._own_id = own_id
        self._batch_size = batch_size

    def get_own_id(self) -> ExecutorId:
        """
        Get the own executor id.

        Returns:
            ExecutorId: The executor id of the executor associated with this
                manager.
        """
        return self._own_id

    def as_executor(self) -> Executor:
        """
        Provide itself as an executor object.

        Returns:
            Executor: The executor object.
        """
        return Executor(self, self._own_id)

    def update_node(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            roa: ReadonlyAccess) -> Node:
        """
        Potentially update the currently active node. If a node change is
        needed or no node was active before the new node is loaded and the
        old node is unloaded.

        Args:
            logger (EventStream): The logger.
            queue_pool (QueuePool): The queue pool.
            roa (ReadonlyAccess): The readonly data access.

        Returns:
            Node: The (new) active node.
        """
        own_id = self._own_id
        node = self._node
        new_node, switch = queue_pool.pick_node(logger, node)
        if switch:
            if node is not None:
                logger.log_event(
                    "tally.node.unload",
                    {
                        "name": "node",
                        "action": "unload",
                        "target": node.get_name(),
                    })
                queue_pool.remove_node_listener(node, own_id)
                node.unload(own_id)
            logger.log_event(
                "tally.node.load",
                {
                    "name": "node",
                    "action": "load",
                    "target": new_node.get_name(),
                })
            queue_pool.add_node_listener(new_node, own_id)
            new_node.load(own_id, roa)
            self._node = new_node
            logger.log_event(
                "tally.node.load",
                {
                    "name": "node",
                    "action": "load_done",
                    "target": new_node.get_name(),
                })
        assert self._node is not None
        return self._node

    def execute_batch(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            store: DataStore,
            roa: ReadonlyAccess) -> bool:
        """
        Execute one batch of tasks. The active node might change based on
        the node strategy. Errors while executing are handled and appropriate
        steps are taken.

        Args:
            logger (EventStream): The logger.
            queue_pool (QueuePool): The queue pool.
            store (DataStore): The payload data store.
            roa (ReadonlyAccess): The readonly data access.

        Returns:
            bool: Whether computation has occurred. If False no tasks were
                available for execution.
        """
        own_id = self._own_id
        node = self.update_node(logger, queue_pool, roa)
        with add_context(node.get_context_info(queue_pool)):
            queue = queue_pool.get_queue(node.get_input_queue())
            tasks = queue.claim_tasks(self._batch_size, own_id)
            logger.log_lazy(
                "debug.task",
                lambda: {
                    "name": "tasks",
                    "tasks": [task.get_task_id() for task in tasks],
                })
            if not tasks:
                return False
            state = ComputeState(queue_pool, store, node, tasks)
            eqids = []
            for qid, (weight, byte_size) in node.expected_meta(state).items():
                queue_pool.expect_task_weight(weight, byte_size, qid, own_id)
                eqids.append(qid)
            e_msg: str | None = None
            r_msg: str | None = None
            e_code: ErrorCode | None = None
            r_code: ErrorCode | None = None
            e_tback: list[str] = []
            r_tback: list[str] = []
            e_pctx: ContextInfo = {}
            r_pctx: ContextInfo = {}
            success = False
            maybe_requeue: dict[TaskId, ErrorInfo] = {}
            try:
                with logger.log_output("output.node", "execute"):
                    node.execute_tasks(state)
                success = True
            except NoTasksToCompute as nttc:
                r_msg = (
                    "all tasks were rejected. "
                    "maybe increase the memory limit "
                    "or reduce the batch size"
                )
                r_code = "memory_purge"
                r_tback = [
                    line.rstrip()
                    for line in traceback.format_exception(nttc)
                ]
                r_pctx = get_preexc_ctx()
            except Exception as exc:  # pylint: disable=broad-except
                e_msg = f"{exc}"
                e_code = "general_exception"
                e_tback = [
                    line.rstrip()
                    for line in traceback.format_exception(exc)
                ]
                e_pctx = get_preexc_ctx()
            for rejected in state.get_current_rejected_tasks():
                pctx: ContextInfo = r_pctx.copy()
                rejected_id = rejected.get_task_id()
                if success:
                    r_msg = "data was purged from memory"
                    r_code = "memory_purge"
                if r_msg is None:
                    r_msg = "unknown error"
                if r_code is None:
                    r_code = "unknown"
                pctx["task"] = rejected_id
                maybe_requeue[rejected_id] = {
                    "ctx": pctx,
                    "message": r_msg,
                    "code": r_code,
                    "traceback": r_tback,
                }
            for qid in eqids:
                queue_pool.clear_expected_task_weight(qid, own_id)
            if success:
                for task in state.results():
                    with add_context({"task": task.get_task_id()}):
                        queue_pool.handle_task_result(logger, store, task)
            else:
                if e_msg is None:
                    e_msg = "unknown error"
                if e_code is None:
                    e_code = "unknown"

                def get_error(task_id: TaskId) -> ErrorInfo:
                    pctx = e_pctx.copy()
                    pctx["task"] = task_id
                    return {
                        "ctx": pctx,
                        "message": e_msg,
                        "code": e_code,
                        "traceback": e_tback,
                    }

                for task in state.get_inputs_tasks():
                    i_task_id = task.get_task_id()
                    if i_task_id not in maybe_requeue:
                        maybe_requeue[i_task_id] = get_error(i_task_id)
            for task_id, error in maybe_requeue.items():
                with add_context({"task": task_id}):
                    queue_pool.maybe_requeue_task_id(
                        logger, store, task_id, error)
            queue.unclaim_tasks(own_id)
            return True

    def reclaim_inactive_tasks(  # FIXME write test for dead executors
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            store: DataStore) -> tuple[int, int]:
        """
        Reclaim tasks from inactive executors. Tasks whose execution was not
        complete for their current node (results were not committed) are made
        available in their queue again.

        Args:
            logger (EventStream): The logger.
            queue_pool (QueuePool): The queue pool.
            store (DataStore): The payload data store.

        Returns:
            tuple[int, int]: The number of reclaimed executors. The first
                number is the number of detected inactive executors. The second
                number is the number of unknown / inactive listeners.
        """
        executor_count = 0
        for executor in self.get_all_executors():
            if executor.is_active():
                continue
            self.handle_inactive_executor(logger, queue_pool, store, executor)
            executor_count += 1

        def is_active(executor_id: ExecutorId) -> bool:
            return self.is_active(executor_id)

        listener_count = queue_pool.clean_listeners(is_active)
        return executor_count, listener_count

    def handle_inactive_executor(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            store: DataStore,
            executor: Executor) -> None:
        """
        Unclaims tasks from an unresponsive executor and makes them available
        in their queue again. This increases the retries count of the tasks.

        Args:
            logger (EventStream): The logger.
            queue_pool (QueuePool): The queue pool.
            store (DataStore): The payload data store.
            executor (Executor): The unresponsive executor.
        """
        executor_id = executor.get_id()
        with add_context({"executor": executor_id}):
            for qid in queue_pool.get_all_queues():
                # NOTE: node listeners are cleaned up separately
                queue_pool.clear_expected_task_weight(qid, executor_id)
                queue = queue_pool.get_queue(qid)
                for reclaim_id in queue.unclaim_tasks(executor_id):
                    with add_context({"task": reclaim_id}):
                        queue_pool.maybe_requeue_task_id(
                            logger,
                            store,
                            reclaim_id,
                            {
                                "ctx": get_ctx(),
                                "message": "executor is unresponsive",
                                "code": "defunc_executor",
                                "traceback": [],
                            })
        executor.release()

    def release_all(self, *, timeout: float | None = None) -> None:
        """
        Release all executors. This usually results in halting all execution.

        Args:
            timeout (float | None, optional): Wait until all executors are
                inactive or the number of seconds runs out. If None the
                function returns immediately in any case. Defaults to None.
        """
        for executor in self.get_all_executors():
            executor.release()
        if timeout is None:
            return
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if not self.any_active():
                return
            sleep_time = timeout - (time.monotonic() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def any_active(self) -> bool:
        """
        Whether there are any registered active executors.

        Returns:
            bool: True if there is at least one active executor.
        """
        for executor in self.get_all_executors():
            if executor.is_active():
                return True
        return False

    def get_all_executors(self) -> list[Executor]:
        """
        Retrieve all registered executors.

        Returns:
            list[Executor]: The executors.
        """
        raise NotImplementedError()

    def is_active(self, executor_id: ExecutorId) -> bool:
        """
        Whether an executor is active.

        Args:
            executor_id (ExecutorId): The executor id.

        Returns:
            bool: True, if the executor is active.
        """
        raise NotImplementedError()

    def release_executor(self, executor_id: ExecutorId) -> None:
        """
        Release an executor and let it stop processing tasks. This does not
        necessarily result in an immediate termination of the associated
        compute resource.

        Args:
            executor_id (ExecutorId): The executor id to stop.
        """
        raise NotImplementedError()

    def start_reclaimer(
            self,
            logger: EventStream,
            reclaim_all_once: Callable[[], tuple[int, int]]) -> None:
        """
        Starts the reclaim loop in the background. At least one reclaim loop
        should be active at any time for non-local workers. The implementation
        must loop indefinitely but can have a long period (i.e., wait time
        between runs).

        Args:
            logger (EventStream): The logger.
            reclaim_all_once (Callable[[], tuple[int, int]]): Reclaims inactive
                executors. This callback cleans up and reclaims all currently
                inactive executors. The callback returns once it is done. It
                does not loop. The numbers returned are the inactive executors
                and the inactive listeners respectively.
        """
        raise NotImplementedError()

    def execute(
            self,
            logger: EventStream,
            work: Callable[['ExecutorManager'], bool]) -> None:
        """
        At its core, this function calls `work` repeatedly until `work` returns
        True. The function might also do bookkeeping and setting the "active"
        status of the own executor. That said, this function might also just
        kick off executing the `work` callback. Furthermore, an executor is
        free to continue working even after `work` return True (ideally, add
        a small timeout before continuing to not spinwait on tasks). For
        stopping executors use the `release_executor` function.

        Args:
            logger (EventStream): The logger.
            work (Callable[[ExecutorManager], bool]): The work. Call this
                function until it returns True (i.e., work is done).
        """
        raise NotImplementedError()

    @staticmethod
    def allow_parallel() -> bool:
        """
        Whether creating multiple executors to achieve local parallelism is
        supported. The call to ::py:method:`execute` happens only to one
        executor in this case so the executor manager needs to ensure that it
        is properly propagated to the other local executors.

        Returns:
            bool: True, if local parallelism is supported.
        """
        raise NotImplementedError()
