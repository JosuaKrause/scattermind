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
"""Configurations connect modules together to make scattermind work."""
import time
from collections.abc import Iterable
from typing import cast, TypeVar

from scattermind.system.base import L_EITHER, Locality, Module, TaskId
from scattermind.system.client.client import ClientPool
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.graphdef import FullGraphDefJSON, json_to_graph
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.logger.log import EventStream
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.queue.queue import (
    NodeStrategy,
    QueuePool,
    QueueStrategy,
)
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.readonly.writer import RoAWriter
from scattermind.system.response import (
    ResponseObject,
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TaskStatus,
)


ModuleT = TypeVar('ModuleT', bound=Module)
"""A module type."""


class Config:
    """Configurations connect different modules together and ensures their
    compatibility. This class also serves as somewhat external API."""
    def __init__(self) -> None:
        """
        Create an empty configuration.
        """
        self._locality: Locality = L_EITHER
        self._logger: EventStream | None = None
        self._graph: Graph | None = None
        self._emng: ExecutorManager | None = None
        self._store: DataStore | None = None
        self._queue_pool: QueuePool | None = None
        self._roa: ReadonlyAccess | None = None

    def set_logger(self, logger: EventStream) -> None:
        """
        Set the logger.

        Args:
            logger (EventStream): The logger.

        Raises:
            ValueError: If the logger is already set.
        """
        if self._logger is not None:
            raise ValueError("logger already initialized")
        self._logger = logger

    def get_logger(self) -> EventStream:
        """
        Get the logger.

        Raises:
            ValueError: If no logger is set.

        Returns:
            EventStream: The logger.
        """
        if self._logger is None:
            raise ValueError("logger not initialized")
        return self._logger

    def set_graph(self, graph: Graph) -> None:
        """
        Set the graph.

        Args:
            graph (Graph): The graph.

        Raises:
            ValueError: If the graph is already set.
        """
        if self._graph is not None:
            raise ValueError("graph already initialized")
        self._graph = graph

    def get_graph(self) -> Graph:
        """
        Get the graph.

        Raises:
            ValueError: If no graph is set.

        Returns:
            Graph: The graph.
        """
        if self._graph is None:
            raise ValueError("graph not initialized")
        return self._graph

    def _update_locality(self, module: ModuleT) -> ModuleT:
        locality = module.locality()
        if self._locality == L_EITHER:
            self._locality = locality
        elif locality == L_EITHER:
            pass
        elif self._locality != locality:  # FIXME write testcase for this
            raise ValueError("trying to load both local and remote modules")
        return module

    def set_executor_manager(self, emng: ExecutorManager) -> None:
        """
        Set the executor manager.

        Args:
            emng (ExecutorManager): The executor manager.

        Raises:
            ValueError: If the executor manager is already set.
        """
        if self._emng is not None:
            raise ValueError("executor manager already initialized")
        self._emng = self._update_locality(emng)

    def get_executor_manager(self) -> ExecutorManager:
        """
        Get the executor manager.

        Raises:
            ValueError: If no executor manager is set.

        Returns:
            ExecutorManager: The executor manager.
        """
        if self._emng is None:
            raise ValueError("executor manager not initialized")
        return self._emng

    def set_data_store(self, store: DataStore) -> None:
        """
        Set the payload data store.

        Args:
            store (DataStore): The payload data store.

        Raises:
            ValueError: If the store is already set.
        """
        if self._store is not None:
            raise ValueError("store already initialized")
        self._store = self._update_locality(store)

    def get_data_store(self) -> DataStore:
        """
        Get the payload data store.

        Raises:
            ValueError: If no store is set.

        Returns:
            DataStore: The payload data store.
        """
        if self._store is None:
            raise ValueError("store not initialized")
        return self._store

    def set_client_pool(self, client_pool: ClientPool) -> None:
        """
        Set the client pool.

        Args:
            client_pool (ClientPool): The client pool.

        Raises:
            ValueError: If the client pool is already set.
        """
        self.get_queue_pool().set_client_pool(
            self._update_locality(client_pool))

    def get_client_pool(self) -> ClientPool:
        """
        Get the client pool.

        Raises:
            ValueError: If no client pool is set.

        Returns:
            ClientPool: The client pool.
        """
        return self.get_queue_pool().get_client_pool()

    def set_queue_pool(self, queue_pool: QueuePool) -> None:
        """
        Set the queue pool.

        Args:
            queue_pool (QueuePool): The queue pool.

        Raises:
            ValueError: If the queue pool is already set.
        """
        if self._queue_pool is not None:
            raise ValueError("queue pool already initialized")
        self._queue_pool = self._update_locality(queue_pool)

    def get_queue_pool(self) -> QueuePool:
        """
        Get the queue pool.

        Raises:
            ValueError: If no queue pool is set.

        Returns:
            QueuePool: The queue pool.
        """
        if self._queue_pool is None:
            raise ValueError("queue pool needs to be initialized first")
        return self._queue_pool

    def set_readonly_access(self, roa: ReadonlyAccess) -> None:
        """
        Set the readonly data access.

        Args:
            roa (ReadonlyAccess): The readonly data access.

        Raises:
            ValueError: If the access is already set.
        """
        if self._roa is not None:
            raise ValueError("readonly access already initialized")
        self._roa = self._update_locality(roa)

    def get_readonly_access(self) -> ReadonlyAccess:
        """
        Get the readonly data access.

        Raises:
            ValueError: If the access is not set.

        Returns:
            ReadonlyAccess: The readonly data access.
        """
        if self._roa is None:
            raise ValueError("readonly access needs to be initialized first")
        return self._roa

    def get_roa_writer(self) -> RoAWriter:
        """
        Reinterpret the readonly data access as writer.

        Raises:
            ValueError: If the readonly data access does not support writing.

        Returns:
            RoAWriter: The writer for offline data creation.
        """
        roa = self.get_readonly_access()
        if not isinstance(roa, RoAWriter):  # FIXME write testcase for this
            raise ValueError("cannot make readonly access writable")
        return cast(RoAWriter, roa)

    def set_node_strategy(self, node_strategy: NodeStrategy) -> None:
        """
        Set the node strategy. Strategies can be set multiple times to update
        them.

        Args:
            node_strategy (NodeStrategy): The node strategy.
        """
        self.get_queue_pool().set_node_strategy(node_strategy)

    def get_node_strategy(self) -> NodeStrategy:
        """
        Get the node strategy.

        Returns:
            NodeStrategy: The node strategy.
        """
        return self.get_queue_pool().get_node_strategy()

    def set_queue_strategy(self, queue_strategy: QueueStrategy) -> None:
        """
        Set the queue strategy. Strategies can be set multiple times to
        update them.

        Args:
            queue_strategy (QueueStrategy): The queue strategy.
        """
        self.get_queue_pool().set_queue_strategy(queue_strategy)

    def get_queue_strategy(self) -> QueueStrategy:
        """
        Get the queue strategy.

        Returns:
            QueueStrategy: The queue strategy.
        """
        return self.get_queue_pool().get_queue_strategy()

    def load_graph(self, graph_def: FullGraphDefJSON) -> None:
        """
        Load the full graph from a JSON definition.

        Args:
            graph_def (FullGraphDefJSON): The JSON definition of the graph.
        """
        graph = json_to_graph(self.get_queue_pool(), graph_def)
        self.set_graph(graph)

    def enqueue(self, value: TaskValueContainer) -> TaskId:
        """
        Enqueues a task.

        Args:
            value (TaskValueContainer): The task's input values.

        Returns:
            TaskId: The task id.
        """
        store = self.get_data_store()
        queue_pool = self.get_queue_pool()
        return queue_pool.enqueue_task(store, value)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """
        Get the status of the given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            TaskStatus: The status of the task.
        """
        cpool = self.get_client_pool()
        return cpool.get_status(task_id)

    def get_result(self, task_id: TaskId) -> TaskValueContainer | None:
        """
        Retrieve the results of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            TaskValueContainer | None: The results of the task or None if no
                results are available.
        """
        cpool = self.get_client_pool()
        queue_pool = self.get_queue_pool()
        graph_id = queue_pool.get_entry_graph()
        output_format = queue_pool.get_output_format(graph_id)
        return cpool.get_final_output(task_id, output_format)

    def get_response(self, task_id: TaskId) -> ResponseObject:
        """
        Retrieve the response of a task. The response gives a summary of
        various elements of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            ResponseObject: The response.
        """
        cpool = self.get_client_pool()
        queue_pool = self.get_queue_pool()
        graph_id = queue_pool.get_entry_graph()
        output_format = queue_pool.get_output_format(graph_id)
        return cpool.get_response(task_id, output_format)

    def clear_task(self, task_id: TaskId) -> None:
        """
        Remove all data associated with the task.

        Args:
            task_id (TaskId): The task id.
        """
        cpool = self.get_client_pool()
        cpool.clear_task(task_id)

    def run(self) -> None:
        """
        Run the executor given by this configuration.
        """
        executor_manager = self.get_executor_manager()
        queue_pool = self.get_queue_pool()
        store = self.get_data_store()
        roa = self.get_readonly_access()
        logger = self.get_logger()

        def work(emng: ExecutorManager) -> bool:
            return emng.execute_batch(logger, queue_pool, store, roa)

        executor_manager.execute(logger, work)

    def wait_for(
            self,
            task_ids: list[TaskId],
            *,
            timeinc: float = 1.0,
            timeout: float = 10.0,
            ) -> Iterable[tuple[TaskId, ResponseObject]]:
        """
        Wait for a collection of tasks to complete.

        Args:
            task_ids (list[TaskId]): The tasks to wait for.
            timeinc (float, optional): The increment of internal waiting
                between checks. Defaults to 1.0.
            timeout (float, optional): The maximum time to wait for any task
                to complete. Defaults to 10.0.

        Yields:
            tuple[TaskId, ResponseObject]: Whenever the next task finishes.
                If the wait times out all remaining tasks are returned (which
                will have an in-progress status). A tuple of task id and its
                response.
        """
        assert timeinc > 0.0
        assert timeout > 0.0
        cur_ids = list(task_ids)
        already: set[TaskId] = set()
        start_time = time.monotonic()
        while cur_ids:
            task_id = cur_ids.pop(0)
            status = self.get_status(task_id)
            print(f"{ctx_fmt()} wait for {task_id} {status}")
            if status in (
                    TASK_STATUS_READY,
                    TASK_STATUS_DONE,
                    TASK_STATUS_ERROR):
                yield (task_id, self.get_response(task_id))
                start_time = time.monotonic()
                continue
            cur_ids.append(task_id)
            if task_id not in already:
                already.add(task_id)
                continue
            already.clear()
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                break
            if elapsed + timeinc > timeout:
                wait_time = timeout - elapsed
            else:
                wait_time = timeinc
            if wait_time > 0.0:
                time.sleep(wait_time)
        for task_id in cur_ids:  # FIXME write timeout test?
            yield (task_id, self.get_response(task_id))
