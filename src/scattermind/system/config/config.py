# Copyright (C) 2024 Josua Krause
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configurations connect modules together to make scattermind work."""
import threading
import time
from collections.abc import Callable, Iterable
from typing import cast, TypeVar

from scattermind.api.api import QueueCounts, ScattermindAPI
from scattermind.system.base import (
    L_EITHER,
    Locality,
    Module,
    SessionId,
    TaskId,
    UserId,
)
from scattermind.system.cache.cache import GraphCache
from scattermind.system.client.client import ClientPool
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.graphdef import FullGraphDefJSON, json_to_graph
from scattermind.system.info import DataFormat
from scattermind.system.logger.log import EventStream
from scattermind.system.names import GNamespace
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
    TASK_COMPLETE,
    TASK_STATUS_DEFER,
    TaskStatus,
)
from scattermind.system.session.session import Session, SessionStore
from scattermind.system.torch_util import DTypeName


ModuleT = TypeVar('ModuleT', bound=Module)
"""A module type."""


class Config(ScattermindAPI):
    """Configurations connect different modules together and ensures their
    compatibility. This class also serves as somewhat external API."""
    def __init__(self) -> None:
        """
        Create an empty configuration.
        """
        self._locality: Locality = L_EITHER
        self._logger: EventStream | None = None
        self._graphs: dict[GNamespace, Graph] = {}
        self._emng: ExecutorManager | None = None
        self._store: DataStore | None = None
        self._queue_pool: QueuePool | None = None
        self._roa: ReadonlyAccess | None = None
        self._sessions: SessionStore | None = None
        self._healthcheck: tuple[str, str, int] | None = None

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

    def add_graph(self, graph: Graph) -> GNamespace:
        """
        Add a graph.

        Args:
            graph (Graph): The graph.

        Raises:
            ValueError: If the graph is already added.

        Returns:
            GNamespace: The namespace of the graph.
        """
        ns = graph.get_namespace()
        if self._graphs.get(ns) is not None:
            raise ValueError(f"graph {ns} already added")
        self._graphs[ns] = graph
        return ns

    def get_graph(self, ns: GNamespace) -> Graph:
        """
        Get the graph of the namespace.

        Args:
            ns (GNamespace): The namespace.

        Raises:
            ValueError: If no graph is set.

        Returns:
            Graph: The graph.
        """
        graph = self._graphs.get(ns)
        if graph is None:
            raise ValueError(f"graph {ns} not initialized")
        return graph

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

    def is_api(self) -> bool:
        """
        Whether this config is an API and cannot be used to execute tasks.

        Returns:
            bool: True, if no executor manager is set.
        """
        return self._emng is None

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

    def set_session_store(self, sessions: SessionStore | None) -> None:
        """
        Set the session store. Session stores are optional. However, if they
        are not set, sessions cannot be used during execution.

        Args:
            sessions (SessionStore | None): The session store or None.
        """
        if sessions is not None:
            sessions = self._update_locality(sessions)
        self._sessions = sessions

    def get_session_store(self) -> SessionStore | None:
        """
        Get the session store. Session stores are optional. However, if they
        are not set, sessions cannot be used during execution.

        Returns:
            SessionStore | None: _description_
        """
        return self._sessions

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

    def set_graph_cache(self, graph_cache: GraphCache) -> None:
        """
        Sets the cache for graph inputs.

        Args:
            graph_cache (GraphCache): The graph cache.
        """
        self.get_queue_pool().set_graph_cache(
            self._update_locality(graph_cache))

    def get_graph_cache(self) -> GraphCache:
        """
        Get the cache for graph inputs.

        Returns:
            GraphCache: The graph cache.
        """
        return self.get_queue_pool().get_graph_cache()

    def set_healthcheck(self, addr_in: str, addr_out: str, port: int) -> None:
        """
        Sets the address at which a healthcheck is exposed.

        Args:
            addr_in (str): The address to listen to the healthcheck
                (e.g., "worker").
            addr_out (str): The address to expose the healthcheck to
                (e.g., "0.0.0.0").
            port (int): The port.
        """
        self._healthcheck = (addr_in, addr_out, port)

    def get_healthcheck(self) -> tuple[str, str, int] | None:
        return self._healthcheck

    def load_graph(self, graph_def: FullGraphDefJSON) -> GNamespace:
        graph = json_to_graph(self.get_queue_pool(), graph_def)
        return self.add_graph(graph)

    def enqueue(
            self,
            ns: GNamespace,
            value: TaskValueContainer,
            *,
            task_id: TaskId | None = None) -> TaskId:
        store = self.get_data_store()
        queue_pool = self.get_queue_pool()
        return queue_pool.enqueue_task(ns, store, value, task_id=task_id)

    def get_namespace(self, task_id: TaskId) -> GNamespace | None:
        cpool = self.get_client_pool()
        return cpool.get_namespace(task_id)

    def wait_for(
            self,
            task_ids: list[TaskId],
            *,
            timeout: float | None = 10.0,
            auto_clear: bool = False,
            ) -> Iterable[tuple[TaskId, ResponseObject]]:
        assert timeout is None or timeout > 0.0
        already_cleared: set[TaskId] = set()
        cpool = self.get_client_pool()
        cur_ids: set[TaskId] = set(task_ids)
        start_time = time.monotonic()
        individual_timeout: float = 60.0 if timeout is None else timeout
        try:
            while cur_ids:
                task_id = cpool.wait_for_task_notifications(
                    list(cur_ids), timeout=individual_timeout)
                elapsed = time.monotonic() - start_time
                if task_id is None:
                    if timeout is not None and elapsed >= timeout:
                        break
                    continue
                status = self.get_status(task_id)
                if status in TASK_COMPLETE:
                    yield (task_id, self.get_response(task_id))
                    start_time = time.monotonic()
                    cur_ids.remove(task_id)
                    if auto_clear:
                        self.clear_task(task_id)
                        already_cleared.add(task_id)
            for task_id in cur_ids:  # FIXME write timeout test?
                yield (task_id, self.get_response(task_id))
                if auto_clear:
                    self.clear_task(task_id)
                    already_cleared.add(task_id)
        finally:
            if auto_clear:
                for task_id in task_ids:
                    if task_id not in already_cleared:
                        self.clear_task(task_id)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        # FIXME: remove side-effects of this method
        cpool = self.get_client_pool()
        res = cpool.get_task_status(task_id)
        if res == TASK_STATUS_DEFER:
            defer_id = cpool.get_deferred_task(task_id)
            if defer_id is not None and defer_id != task_id:
                defer_status = self.get_status(defer_id)
                if defer_status not in TASK_COMPLETE:
                    return defer_status
            cpool.clear_task(task_id)
            logger = self.get_logger()
            store = self.get_data_store()
            queue_pool = self.get_queue_pool()
            queue_pool.maybe_requeue_task_id(
                logger, store, task_id, error_info=None)
            res = cpool.get_task_status(task_id)
        return res

    def get_result(self, task_id: TaskId) -> TaskValueContainer | None:
        cpool = self.get_client_pool()
        ns = cpool.get_namespace(task_id)
        if ns is None:
            return None
        queue_pool = self.get_queue_pool()
        graph_id = queue_pool.get_entry_graph(ns)
        output_format = queue_pool.get_output_format(graph_id)
        return cpool.get_final_output(task_id, output_format)

    def get_response(self, task_id: TaskId) -> ResponseObject:
        cpool = self.get_client_pool()
        ns = cpool.get_namespace(task_id)
        if ns is None:
            return cpool.get_response(task_id, None)
        queue_pool = self.get_queue_pool()
        graph_id = queue_pool.get_entry_graph(ns)
        output_format = queue_pool.get_output_format(graph_id)
        return cpool.get_response(task_id, output_format)

    def clear_task(self, task_id: TaskId) -> None:
        cpool = self.get_client_pool()
        cpool.clear_task(task_id)

    def run(self, *, force_no_block: bool, no_reclaim: bool) -> int | None:
        """
        Run the executor given by this configuration.

        Args:
            force_no_block (bool): If set, forces the function to become
                non-blocking.
            no_reclaim (bool): If set, the reclaimer will not be executed.
                This is only recommended for tests.

        Returns:
            int | None: If the call is blocking (i.e., the work is done inside
                this function call) the exit code is returned as int. Otherwise
                the function returns None.
        """
        executor_manager = self.get_executor_manager()
        cpool = self.get_client_pool()
        queue_pool = self.get_queue_pool()
        store = self.get_data_store()
        roa = self.get_readonly_access()
        sessions = self.get_session_store()
        logger = self.get_logger()

        def reclaim_all_once() -> tuple[int, int]:
            return executor_manager.reclaim_inactive_tasks(
                logger, cpool, queue_pool, store)

        if not no_reclaim:
            executor_manager.start_reclaimer(logger, reclaim_all_once)

        def wait_for_task(
                is_release_requested: Callable[[], bool],
                timeout: float) -> None:

            def task_condition() -> bool:
                if is_release_requested():
                    return True
                return self.has_any_tasks()

            cpool.wait_for_queues(task_condition, timeout)

        def work(emng: ExecutorManager) -> bool:
            return emng.execute_batch(logger, queue_pool, store, sessions, roa)

        def do_execute() -> int | None:
            return executor_manager.execute(
                logger,
                wait_for_task=wait_for_task,
                work=work)

        if not force_no_block:
            return do_execute()

        th = threading.Thread(target=do_execute, daemon=False)
        th.start()
        return None

    def namespaces(self) -> set[GNamespace]:
        return set(self._graphs.keys())

    def entry_graph_name(self, ns: GNamespace) -> str:
        queue_pool = self.get_queue_pool()
        return queue_pool.get_graph_name(
            queue_pool.get_entry_graph(ns)).to_parseable()

    def main_input_format(self, ns: GNamespace) -> DataFormat:
        queue_pool = self.get_queue_pool()
        return queue_pool.get_input_format(queue_pool.get_entry_graph(ns))

    def main_outputs(self, ns: GNamespace) -> set[str]:
        queue_pool = self.get_queue_pool()
        output = queue_pool.get_output_format(queue_pool.get_entry_graph(ns))
        return set(output.keys())

    def output_format(
            self,
            ns: GNamespace,
            output_name: str) -> tuple[DTypeName, list[int | None]]:
        queue_pool = self.get_queue_pool()
        output = queue_pool.get_output_format(queue_pool.get_entry_graph(ns))
        output_fmt = output[output_name]
        return output_fmt.dtype(), output_fmt.shape()

    def get_queue_stats(
            self, ns: GNamespace | None = None) -> Iterable[QueueCounts]:
        queue_pool = self.get_queue_pool()
        for qid in queue_pool.get_all_queues():
            queue = queue_pool.get_queue(qid)
            node = queue.get_consumer_node()
            graph_name = queue_pool.get_graph_name(node.get_graph())
            if ns is not None and graph_name.get_namespace() != ns:
                continue
            listerners = queue_pool.get_queue_listeners(qid)
            queue_length = queue.get_queue_length()
            if listerners <= 0 and queue_length <= 0:
                continue
            qual_name = node.get_qualified_name(queue_pool)
            yield {
                "id": qid,
                "name": qual_name,
                "queue_length": queue_length,
                "listeners": listerners,
            }

    def has_any_tasks(self) -> bool:
        queue_pool = self.get_queue_pool()
        for qid in queue_pool.get_all_queues():
            queue = queue_pool.get_queue(qid)
            queue_length = queue.get_queue_length()
            if queue_length > 0:
                return True
        return False

    def new_session(
            self,
            user_id: UserId,
            *,
            copy_from: Session | None = None) -> Session:
        sessions = self.get_session_store()
        if sessions is None:
            raise ValueError("no session store defined")
        copy_id = None if copy_from is None else copy_from.get_session_id()
        return sessions.create_new_session(user_id, copy_from=copy_id)

    def get_session(self, session_id: SessionId) -> Session:
        sessions = self.get_session_store()
        if sessions is None:
            raise ValueError("no session store defined")
        return sessions.get_session(session_id)

    def get_session_user(self, session_id: SessionId) -> UserId | None:
        sessions = self.get_session_store()
        if sessions is None:
            raise ValueError("no session store defined")
        return sessions.get_user(session_id)

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        sessions = self.get_session_store()
        if sessions is None:
            yield from []
        else:
            yield from sessions.get_sessions(user_id)
