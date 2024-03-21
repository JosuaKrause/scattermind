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
from collections.abc import Iterable
from typing import cast, TypeVar

from scattermind.api.api import QueueCounts, ScattermindAPI
from scattermind.system.base import L_EITHER, Locality, Module, TaskId
from scattermind.system.cache.cache import GraphCache
from scattermind.system.client.client import ClientPool
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.graphdef import FullGraphDefJSON, json_to_graph
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
from scattermind.system.response import ResponseObject, TaskStatus
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

    def enqueue(self, ns: GNamespace, value: TaskValueContainer) -> TaskId:
        store = self.get_data_store()
        queue_pool = self.get_queue_pool()
        return queue_pool.enqueue_task(ns, store, value)

    def get_namespace(self, task_id: TaskId) -> GNamespace | None:
        cpool = self.get_client_pool()
        return cpool.get_namespace(task_id)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        cpool = self.get_client_pool()
        return cpool.get_status(task_id)

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

    def run(self, *, force_no_block: bool) -> int | None:
        """
        Run the executor given by this configuration.

        Args:
            force_no_block (bool): If set, forces the function to become
                non-blocking.

        Returns:
            int | None: If the call is blocking (i.e., the work is done inside
                this function call) the exit code is returned as int. Otherwise
                the function returns None.
        """
        executor_manager = self.get_executor_manager()
        queue_pool = self.get_queue_pool()
        store = self.get_data_store()
        roa = self.get_readonly_access()
        logger = self.get_logger()

        def reclaim_all_once() -> tuple[int, int]:
            return executor_manager.reclaim_inactive_tasks(
                logger, queue_pool, store)

        executor_manager.start_reclaimer(logger, reclaim_all_once)

        def work(emng: ExecutorManager) -> bool:
            return emng.execute_batch(logger, queue_pool, store, roa)

        def do_execute() -> int | None:
            return executor_manager.execute(logger, work)

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

    def main_inputs(self, ns: GNamespace) -> set[str]:
        queue_pool = self.get_queue_pool()
        inputs = queue_pool.get_input_format(queue_pool.get_entry_graph(ns))
        return set(inputs.keys())

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

    def get_queue_stats(self) -> Iterable[QueueCounts]:
        queue_pool = self.get_queue_pool()
        for qid in queue_pool.get_all_queues():
            queue = queue_pool.get_queue(qid)
            listerners = queue_pool.get_queue_listeners(qid)
            queue_length = queue.get_queue_length()
            if listerners <= 0 and queue_length <= 0:
                continue
            qual_name = queue.get_consumer_node().get_qualified_name(
                queue_pool)
            yield {
                "id": qid,
                "name": qual_name,
                "queue_length": queue_length,
                "listeners": listerners,
            }
