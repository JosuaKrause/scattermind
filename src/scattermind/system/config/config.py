import time
from collections.abc import Iterable
from typing import cast, TypeVar

from scattermind.system.base import Module, TaskId
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


class Config:
    def __init__(self) -> None:
        self._is_local: bool | None = None
        self._logger: EventStream | None = None
        self._graph: Graph | None = None
        self._emng: ExecutorManager | None = None
        self._store: DataStore | None = None
        self._queue_pool: QueuePool | None = None
        self._roa: ReadonlyAccess | None = None

    def set_logger(self, logger: EventStream) -> None:
        if self._logger is not None:
            raise ValueError("logger already initialized")
        self._logger = logger

    def get_logger(self) -> EventStream:
        if self._logger is None:
            raise ValueError("logger not initialized")
        return self._logger

    def set_graph(self, graph: Graph) -> None:
        if self._graph is not None:
            raise ValueError("graph already initialized")
        self._graph = graph

    def get_graph(self) -> Graph:
        if self._graph is None:
            raise ValueError("graph not initialized")
        return self._graph

    def _update_is_local(self, module: ModuleT) -> ModuleT:
        is_local = module.is_local_only()
        if self._is_local is None:
            self._is_local = is_local
        elif self._is_local != is_local:  # FIXME write testcase for this
            raise ValueError("trying to load local and non-local modules")
        return module

    def set_executor_manager(self, emng: ExecutorManager) -> None:
        if self._emng is not None:
            raise ValueError("executor manager already initialized")
        self._emng = self._update_is_local(emng)

    def get_executor_manager(self) -> ExecutorManager:
        if self._emng is None:
            raise ValueError("executor manager not initialized")
        return self._emng

    def set_data_store(self, store: DataStore) -> None:
        if self._store is not None:
            raise ValueError("store already initialized")
        self._store = self._update_is_local(store)

    def get_data_store(self) -> DataStore:
        if self._store is None:
            raise ValueError("store not initialized")
        return self._store

    def set_client_pool(self, client_pool: ClientPool) -> None:
        self.get_queue_pool().set_client_pool(
            self._update_is_local(client_pool))

    def get_client_pool(self) -> ClientPool:
        return self.get_queue_pool().get_client_pool()

    def set_queue_pool(self, queue_pool: QueuePool) -> None:
        if self._queue_pool is not None:
            raise ValueError("queue pool already initialized")
        self._queue_pool = self._update_is_local(queue_pool)

    def get_queue_pool(self) -> QueuePool:
        if self._queue_pool is None:
            raise ValueError("queue pool needs to be initialized first")
        return self._queue_pool

    def set_readonly_access(self, roa: ReadonlyAccess) -> None:
        if self._roa is not None:
            raise ValueError("readonly access already initialized")
        self._roa = self._update_is_local(roa)

    def get_readonly_access(self) -> ReadonlyAccess:
        if self._roa is None:
            raise ValueError("readonly access needs to be initialized first")
        return self._roa

    def get_roa_writer(self) -> RoAWriter:
        roa = self.get_readonly_access()
        if not isinstance(roa, RoAWriter):  # FIXME write testcase for this
            raise ValueError("cannot make readonly access writable")
        return cast(RoAWriter, roa)

    def set_node_strategy(self, node_strategy: NodeStrategy) -> None:
        self.get_queue_pool().set_node_strategy(node_strategy)

    def get_node_strategy(self) -> NodeStrategy:
        return self.get_queue_pool().get_node_strategy()

    def set_queue_strategy(self, queue_strategy: QueueStrategy) -> None:
        self.get_queue_pool().set_queue_strategy(queue_strategy)

    def get_queue_strategy(self) -> QueueStrategy:
        return self.get_queue_pool().get_queue_strategy()

    def load_graph(self, graph_def: FullGraphDefJSON) -> None:
        graph = json_to_graph(self.get_queue_pool(), graph_def)
        self.set_graph(graph)

    def enqueue(self, value: TaskValueContainer) -> TaskId:
        store = self.get_data_store()
        queue_pool = self.get_queue_pool()
        return queue_pool.enqueue_task(store, value)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        cpool = self.get_client_pool()
        return cpool.get_status(task_id)

    def get_result(self, task_id: TaskId) -> TaskValueContainer | None:
        cpool = self.get_client_pool()
        return cpool.get_final_output(task_id)

    def get_response(self, task_id: TaskId) -> ResponseObject:
        cpool = self.get_client_pool()
        return cpool.get_response(task_id)

    def clear_task(self, task_id: TaskId) -> None:
        cpool = self.get_client_pool()
        cpool.clear_task(task_id)

    def run(self) -> None:
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
