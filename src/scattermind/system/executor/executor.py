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
    def __init__(
            self, emng: 'ExecutorManager', executor_id: ExecutorId) -> None:
        self._emng = emng
        self._executor_id = executor_id

    def get_id(self) -> ExecutorId:
        return self._executor_id

    def is_active(self) -> bool:
        return self._emng.is_active(self._executor_id)

    def release(self) -> None:
        return self._emng.release_executor(self._executor_id)


class ExecutorManager(Module):
    def __init__(self, own_id: ExecutorId, batch_size: int) -> None:
        self._node: Node | None = None
        self._own_id = own_id
        self._batch_size = batch_size

    def get_own_id(self) -> ExecutorId:
        return self._own_id

    def as_executor(self) -> Executor:
        return Executor(self, self._own_id)

    def update_node(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            roa: ReadonlyAccess) -> Node:
        own_id = self._own_id
        node = self._node
        new_node, switch = queue_pool.pick_node(logger, node)
        if switch:
            if node is not None:
                logger.log_event(
                    "tally.node.unload", {"name": "node", "action": "unload"})
                node.unload(own_id)
            logger.log_event(
                "tally.node.load", {"name": "node", "action": "load"})
            new_node.load(own_id, roa)
            self._node = new_node
        assert self._node is not None
        return self._node

    def execute_batch(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            store: DataStore,
            roa: ReadonlyAccess) -> bool:
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
            store: DataStore) -> None:
        for executor in self.get_all_executors():
            if executor.is_active():
                continue
            self.handle_inactive_executor(logger, queue_pool, store, executor)

    def handle_inactive_executor(
            self,
            logger: EventStream,
            queue_pool: QueuePool,
            store: DataStore,
            executor: Executor) -> None:
        executor_id = executor.get_id()
        with add_context({"executor": executor_id}):
            for qid in queue_pool.get_all_queues():
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
        for executor in self.get_all_executors():
            if executor.is_active():
                return True
        return False

    def get_all_executors(self) -> list[Executor]:
        raise NotImplementedError()

    def is_active(self, executor_id: ExecutorId) -> bool:
        raise NotImplementedError()

    def release_executor(self, executor_id: ExecutorId) -> None:
        raise NotImplementedError()

    def execute(
            self,
            logger: EventStream,
            work: Callable[['ExecutorManager'], bool]) -> None:
        raise NotImplementedError()
