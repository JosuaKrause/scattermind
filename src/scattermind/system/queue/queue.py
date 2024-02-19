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
"""Provides the queue and queue pool."""
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from scattermind.system.base import (
    ExecutorId,
    GraphId,
    Module,
    QueueId,
    TaskId,
)
from scattermind.system.client.client import ClientPool, ComputeTask
from scattermind.system.info import DataFormat
from scattermind.system.logger.context import ctx_fmt, get_ctx
from scattermind.system.logger.error import ErrorInfo, to_retry_event
from scattermind.system.logger.event import QueueMeasureEvent
from scattermind.system.logger.log import EventStream
from scattermind.system.names import (
    GNamespace,
    QualifiedGraphName,
    QualifiedName,
    ValueMap,
)
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    PICK_LEFT,
    PICK_RIGHT,
    QueueStrategy,
)
from scattermind.system.response import (
    TASK_STATUS_BUSY,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TASK_STATUS_WAIT,
    TaskStatus,
)


if TYPE_CHECKING:
    from scattermind.system.graph.node import Node


class Queue:
    """Convenient access to queue functionality without having to work with
    queue ids."""
    def __init__(self, queue_pool: 'QueuePool', qid: QueueId) -> None:
        """
        Creates a queue wrapper.

        Args:
            queue_pool (QueuePool): The queue pool.
            qid (QueueId): The queue id.
        """
        self._queue_pool = queue_pool
        self._qid = qid.ensure_queue()

    def get_id(self) -> QueueId:
        """
        Get the queue id.

        Returns:
            QueueId: The queue id.
        """
        return self._qid

    def get_consumer_node(self) -> 'Node':
        """
        Retrieves the node that consumes this queue.

        Returns:
            Node: The node.
        """
        return self._queue_pool.get_consumer_node(self._qid)

    def claimant_count(self) -> int:
        """
        Retrieves the number of executors having claims on this queue.

        Returns:
            int: The number of executors having claims on the queue.
        """
        return self._queue_pool.claimant_count(self._qid)

    def get_listener_count(self) -> int:
        """
        Counts the number of listeners (executors) that have loaded the
        associated node of this queue.

        Returns:
            int: The number of listeners (executors).
        """
        return self._queue_pool.get_queue_listeners(self._qid)

    def get_unclaimed_tasks(self) -> list[ComputeTask]:
        """
        Retrieves all unclaimed tasks in this queue.

        Returns:
            list[ComputeTask]: The list of unclaimed tasks as states.
        """
        return self._queue_pool.get_unclaimed_compute_tasks(self._qid)

    def get_expected_new_task_weight(self) -> float:
        """
        Retrieves all current expected total new task weights of the queue.
        This does not take the actual current total task weight / backpressure
        into account.

        Returns:
            float: The total expected task weight.
        """
        return self._queue_pool.get_expected_new_task_weight(self._qid)

    def get_queue_length(self) -> int:
        """
        Retrieves the current length of the queue.

        Returns:
            int: The length of the queue.
        """
        return self._queue_pool.get_queue_length(self._qid)

    def get_expected_byte_size(self) -> int:
        """
        Retrieves the current expected payload size of the queue.
        This does not take the actual current payload size into account.

        Returns:
            int: The total expected payload size of the queue.
        """
        return self._queue_pool.get_expected_byte_size(self._qid)

    def get_incoming_byte_size(self) -> int:
        """
        Retrieves the actual current payload size of the queue.

        Returns:
            int: The total current payload size of the queue.
        """
        return self._queue_pool.get_incoming_byte_size(self._qid)

    def total_weight(self) -> float:
        """
        Retrieves the actual current weight of the queue without taking the
        byte size into account.

        Returns:
            float: The total task weight currently in the queue.
        """
        res = 0.0
        for task in self.get_unclaimed_tasks():
            res += task.get_simple_weight_in()
        return res

    def total_backpressure(self) -> float:
        """
        Retrieves the actual current backpressure / weight of the queue.

        Returns:
            float: The total weight (pressure) currently in the queue.
        """
        res = 0.0
        for task in self.get_unclaimed_tasks():
            res += task.get_total_weight_in()
        return res

    def comparative_backpressure(self) -> float:
        """
        Computes the current backpressure / weight of the queue.

        Returns:
            float: A comparable backpressure. The number makes only sense in
                comparison to other queues. It does not give an absolute
                estimate of the required work in the queue.
        """
        res = 0.0
        for task in self.get_unclaimed_tasks():
            res += task.get_simple_weight_in()
        return res

    def total_expected_pressure(self) -> float:
        """
        Computes the total expected pressure in the queue.

        Returns:
            float: The total expected pressure.
        """
        total_weight = self.get_expected_new_task_weight()
        return total_weight * self.get_expected_byte_size()

    def comparative_expected_pressure(self) -> float:
        """
        Computes the currently expected pressure in the queue.

        Returns:
            float: A comparable pressure. The number makes only sense in
                comparison to other queues. It does not give an absolute
                estimate of the required expected new work in the queue.
        """
        return self.get_expected_new_task_weight()

    def claim_tasks(
            self,
            batch_size: int,
            executor_id: ExecutorId) -> list[ComputeTask]:
        """
        Claim tasks on the queue for computation.

        Args:
            batch_size (int): The desired number of tasks to claim.
            executor_id (ExecutorId): The executor claiming the tasks.

        Returns:
            list[ComputeTask]: The claimed tasks as state.
        """
        return self._queue_pool.claim_compute_tasks(
            self._qid, batch_size, executor_id)

    def unclaim_tasks(self, executor_id: ExecutorId) -> list[TaskId]:
        """
        Unclaim all tasks of an executor.

        Args:
            executor_id (ExecutorId): The executor.

        Returns:
            list[TaskId]: The tasks that were previously claimed by the
                executor as task id.
        """
        return self._queue_pool.unclaim_tasks(self._qid, executor_id)

    def push_task_id(self, task_id: TaskId) -> None:
        """
        Pushes a task to the queue.

        Args:
            task_id (TaskId): The task id.
        """
        return self._queue_pool.push_task_id(self._qid, task_id)


class QueuePool(Module):
    """
    A queue pool keeps track of all queues. It connects nodes via queues.
    It also maintains information about graph input and output queues and their
    formats. A client pool needs to be connected to the queue pool in order to
    be able to handle tasks.
    """
    def __init__(self) -> None:
        """Create an empty queue pool."""
        super().__init__()
        self._client_pool: ClientPool | None = None
        self._last_entry_graph: GraphId | None = None
        self._entry_graphs: dict[GNamespace, GraphId] = {}
        self._graph_ids: dict[QualifiedGraphName, GraphId] = {}
        self._graph_names: dict[GraphId, QualifiedGraphName] = {}
        self._graph_descs: dict[GraphId, str] = {}
        self._input_nodes: dict[GraphId, 'Node'] = {}
        self._input_formats: dict[GraphId, DataFormat] = {}
        self._output_formats: dict[GraphId, DataFormat] = {}
        self._output_vmaps: dict[GraphId, ValueMap] = {}
        self._nodes: set['Node'] = set()
        self._input_queues: dict[QueueId, 'Node'] = {}
        self._node_strategy: NodeStrategy | None = None
        self._queue_strategy: QueueStrategy | None = None

    def set_client_pool(self, client_pool: ClientPool) -> None:
        """
        Connects the client pool. This can be done only once.

        Args:
            client_pool (ClientPool): The client pool.

        Raises:
            ValueError: If the client pool is already set.
        """
        if self._client_pool is not None:
            raise ValueError("client pool already initialized")
        self._client_pool = client_pool

    def get_client_pool(self) -> ClientPool:
        """
        Retrieve the associated client pool.

        Raises:
            ValueError: If the client pool hasn't been initialized yet.

        Returns:
            ClientPool: The client pool.
        """
        if self._client_pool is None:
            raise ValueError("client pool not initialized")
        return self._client_pool

    def set_entry_graph(self, ns: GNamespace, graph_id: GraphId) -> None:
        """
        Sets the entry graph.

        Args:
            ns (GNamespace): The namespace.
            graph_id (GraphId): The graph that gets the overall input of the
                execution graph.

        Raises:
            ValueError: If the entry graph has already been initialized.
        """
        if self._entry_graphs.get(ns) is not None:
            raise ValueError(f"entry graph already initialized for {ns}")
        self._entry_graphs[ns] = graph_id
        self._last_entry_graph = graph_id

    def get_entry_graph(self, ns: GNamespace) -> GraphId:
        """
        Retrieves the entry graph for the given namespace.

        Args:
            ns (GNamespace): The namespace.

        Raises:
            ValueError: If the entry graph hasn't been set for the namespace.

        Returns:
            GraphId: The graph that gets the overall input of the
                execution graph in the namespace.
        """
        res = self._entry_graphs.get(ns)
        if res is None:
            raise ValueError(f"entry graph not initialized for {ns}")
        return res

    def last_entry_graph(self) -> GraphId:
        """
        Retrieves the last added entry graph.

        Returns:
            GraphId: The graph.
        """
        res = self._last_entry_graph
        if res is None:
            raise ValueError("no entry graph set")
        return res

    def add_graph(
            self,
            graph_id: GraphId,
            gname: QualifiedGraphName,
            desc: str) -> None:
        """
        Initialize a new (sub-)graph. Graphs need to be uniquely identifyable
        via graph id and name.

        Args:
            graph_id (GraphId): The graph id.
            gname (QualifiedGraphName): The qualified graph name.
            desc (str): The description of the graph.

        Raises:
            ValueError: If the graph id or name are ambiguous.
        """
        if graph_id in self._graph_names:
            raise ValueError(f"duplicate graph {graph_id}")
        if gname in self._graph_ids:
            raise ValueError(f"duplicate graph name: {gname.to_parseable()}")
        self._graph_ids[gname] = graph_id
        self._graph_names[graph_id] = gname
        self._graph_descs[graph_id] = desc

    def get_graph_id(self, gname: QualifiedGraphName) -> GraphId:
        """
        Get the graph id for the given name.

        Args:
            gname (QualifiedGraphName): The qualified name of the graph.

        Returns:
            GraphId: The associated graph id.
        """
        return self._graph_ids[gname]

    def get_graph_name(self, graph_id: GraphId) -> QualifiedGraphName:
        """
        Get the qualified graph name for the given id.

        Args:
            graph_id (GraphId): The id of the graph.

        Returns:
            QualifiedGraphName: The associated qualified graph name.
        """
        return self._graph_names[graph_id]

    def get_graphs(self) -> list[GraphId]:
        """
        Gets all (sub-)graphs served by this queue pool.

        Returns:
            list[GraphId]: The list of graph ids.
        """
        return list(self._graph_names.keys())

    def get_graph_description(self, graph_id: GraphId) -> str:
        """
        Retrieve the description of the given graph.

        Args:
            graph_id (GraphId): The graph id.

        Returns:
            str: The description.
        """
        return self._graph_descs[graph_id]

    def set_input_node(self, graph_id: GraphId, node: 'Node') -> None:
        """
        Sets the input node of a graph.

        Args:
            graph_id (GraphId): The graph id.
            node (Node): The input node.

        Raises:
            ValueError: If the input node for the graph is already set.
        """
        if graph_id in self._input_nodes:
            raise ValueError(f"input node for {graph_id} already initialized")
        self._input_nodes[graph_id] = node

    def get_input_node(self, graph_id: GraphId) -> 'Node':
        """
        Retrieve the input node of the given graph.

        Args:
            graph_id (GraphId): The graph id.

        Raises:
            ValueError: If the input node has not been set.

        Returns:
            Node: The input node.
        """
        res = self._input_nodes.get(graph_id)
        if res is None:
            raise ValueError(f"input node for {graph_id} not initialized")
        return res

    def set_input_format(
            self, graph_id: GraphId, input_format: DataFormat) -> None:
        """
        Sets the input format of the graph.

        Args:
            graph_id (GraphId): The graph id.
            input_format (DataFormat): The expected input format.

        Raises:
            ValueError: If the input format has already been defined.
        """
        if graph_id in self._input_formats:
            raise ValueError(
                f"input format for {graph_id} already initialized")
        self._input_formats[graph_id] = input_format

    def get_input_format(self, graph_id: GraphId) -> DataFormat:
        """
        Gets the input format of the graph.

        Args:
            graph_id (GraphId): The graph id.

        Raises:
            ValueError: If the input format has not been defined.

        Returns:
            DataFormat: The expected input format.
        """
        res = self._input_formats.get(graph_id)
        if res is None:
            raise ValueError(f"input format for {graph_id} not initialized")
        return res

    def set_output_format(
            self, graph_id: GraphId, output_format: DataFormat) -> None:
        """
        Set the output format of the graph.

        Args:
            graph_id (GraphId): The graph id.
            output_format (DataFormat): The expected output format.

        Raises:
            ValueError: If the output format has already been set.
        """
        if graph_id in self._output_formats:
            raise ValueError(
                f"output format for {graph_id} already initialized")
        self._output_formats[graph_id] = output_format

    def get_output_format(self, graph_id: GraphId) -> DataFormat:
        """
        Get the output format of the graph.

        Args:
            graph_id (GraphId): The graph id.

        Raises:
            ValueError: If the output format has not been set.

        Returns:
            DataFormat: The expected output format.
        """
        res = self._output_formats.get(graph_id)
        if res is None:
            raise ValueError(f"output format for {graph_id} not initialized")
        return res

    def set_output_value_map(self, graph_id: GraphId, vmap: ValueMap) -> None:
        """
        Set the output map which maps the output of nodes in the graph to
        fields in the output of the graph.

        Args:
            graph_id (GraphId): The graph id.
            vmap (ValueMap): The output map.

        Raises:
            ValueError: If the output map has already been set.
        """
        if graph_id in self._output_vmaps:
            raise ValueError("output value map already initialized")
        self._output_vmaps[graph_id] = vmap

    def get_output_value_map(self, graph_id: GraphId) -> ValueMap:
        """
        Retrieve the output map which maps the output of nodes in the graph to
        fields in the output of the graph.

        Args:
            graph_id (GraphId): The graph id.

        Returns:
            ValueMap: The output map.
        """
        return self._output_vmaps[graph_id]

    def set_node_strategy(self, node_strategy: NodeStrategy) -> None:
        """
        Sets the strategy for choosing which node to process next.

        Args:
            node_strategy (NodeStrategy): The strategy.
        """
        self._node_strategy = node_strategy

    def get_node_strategy(self) -> NodeStrategy:
        """
        Gets the current strategy for choosing which node to process next.

        Raises:
            ValueError: If no strategy has been set.

        Returns:
            NodeStrategy: The strategy.
        """
        if self._node_strategy is None:
            raise ValueError("node strategy not set!")
        return self._node_strategy

    def set_queue_strategy(self, queue_strategy: QueueStrategy) -> None:
        """
        Sets the strategy for choosing which task to process next for a given
        queue.

        Args:
            queue_strategy (QueueStrategy): The strategy.
        """
        self._queue_strategy = queue_strategy

    def get_queue_strategy(self) -> QueueStrategy:
        """
        Gets the strategy for choosing which task to process next for a given
        queue.

        Raises:
            ValueError: If the strategy has not been set.

        Returns:
            QueueStrategy: The strategy.
        """
        if self._queue_strategy is None:
            raise ValueError("queue strategy not set!")
        return self._queue_strategy

    def get_all_nodes(self) -> Iterable['Node']:
        """
        Get all nodes registered in the queue pool.

        Yields:
            Node: The node.
        """
        yield from self._nodes

    def register_node(self, node: 'Node') -> None:
        """
        Register a node with the queue pool.

        Args:
            node (Node): The node.

        Raises:
            ValueError: If the input queue id of the node exists already
                in the pool.
        """
        qid = node.get_input_queue()
        if qid in self._input_queues:
            raise ValueError(
                f"duplicate input queue id: {qid} already registered "
                f"for {self._input_queues[qid]}")
        self._input_queues[qid] = node
        self._nodes.add(node)

    def get_queue(self, qid: QueueId) -> Queue:
        """
        Retrieve the queue associated with the given id.

        Args:
            qid (QueueId): The queue id.

        Returns:
            Queue: The queue object.
        """
        return Queue(self, qid)

    def get_consumer_node(self, qid: QueueId) -> 'Node':
        """
        Retrieve the node consuming the given queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            Node: The node consuming the queue.
        """
        return self._input_queues[qid]

    def get_all_queues(self) -> Iterable[QueueId]:
        """
        Return all registered queues.

        Yields:
            QueueId: The queue id.
        """
        yield from self._input_queues

    def pick_node(
            self,
            logger: EventStream,
            current_node: 'Node | None') -> tuple['Node', bool]:
        """
        Pick a node to process next.

        Args:
            logger (EventStream): The logger.
            current_node (Node | None): The previously processed node or None
                if no node has been processed yet.

        Returns:
            tuple[Node, bool]: A tuple of the next node to process (or None if
                no processing is needed at the moment) and a boolean that if
                True indicates that the node has changed wrt. the previous
                node. In that case the previous node needs to be unloaded and
                the new node needs to be loaded.
        """
        strategy = self.get_node_strategy()
        last_entry_graph_id = self.last_entry_graph()
        candidate_node = self.get_input_node(last_entry_graph_id)

        def process_node(left_node: 'Node', right_node: 'Node') -> 'Node':
            left_graph_id = left_node.get_graph()
            left_graph_name = self.get_graph_name(left_graph_id)
            left_qid = left_node.get_input_queue()
            left_queue = self.get_queue(left_qid)
            right_graph_id = right_node.get_graph()
            right_graph_name = self.get_graph_name(right_graph_id)
            right_qid = right_node.get_input_queue()
            right_queue = self.get_queue(right_qid)
            left_qme: QueueMeasureEvent = {
                "name": "queue_input",
            }
            right_qme: QueueMeasureEvent = {
                "name": "queue_input",
            }

            def left_queue_length() -> int:
                queue_length = left_queue.get_queue_length()
                left_qme["length"] = queue_length
                return queue_length

            def left_weight() -> float:
                weight = left_queue.total_weight()
                left_qme["weight"] = weight
                return weight

            def left_pressure() -> float:
                pressure = left_queue.total_backpressure()
                left_qme["pressure"] = pressure
                return pressure

            def left_expected_pressure() -> float:
                expected_pressure = left_queue.total_expected_pressure()
                left_qme["expected_pressure"] = expected_pressure
                return expected_pressure

            def left_cost_to_load() -> float:
                cost_to_load = left_node.get_load_cost()
                left_qme["cost"] = cost_to_load
                return cost_to_load

            def left_claimants() -> int:
                claimants = left_queue.claimant_count()
                left_qme["claimants"] = claimants
                return claimants

            def left_loaded() -> int:
                loaded = left_queue.get_listener_count()
                left_qme["loaded"] = loaded
                return loaded

            def right_queue_length() -> int:
                queue_length = right_queue.get_queue_length()
                right_qme["length"] = queue_length
                return queue_length

            def right_weight() -> float:
                weight = right_queue.total_weight()
                right_qme["weight"] = weight
                return weight

            def right_pressure() -> float:
                pressure = right_queue.total_backpressure()
                right_qme["pressure"] = pressure
                return pressure

            def right_expected_pressure() -> float:
                expected_pressure = right_queue.total_expected_pressure()
                right_qme["expected_pressure"] = expected_pressure
                return expected_pressure

            def right_cost_to_load() -> float:
                cost_to_load = right_node.get_load_cost()
                right_qme["cost"] = cost_to_load
                return cost_to_load

            def right_claimants() -> int:
                claimants = right_queue.claimant_count()
                right_qme["claimants"] = claimants
                return claimants

            def right_loaded() -> int:
                loaded = right_queue.get_listener_count()
                right_qme["loaded"] = loaded
                return loaded

            pick_node = strategy.pick_node(
                left_queue_length=left_queue_length,
                left_weight=left_weight,
                left_pressure=left_pressure,
                left_expected_pressure=left_expected_pressure,
                left_cost_to_load=left_cost_to_load,
                left_claimants=left_claimants,
                left_loaded=left_loaded,
                right_queue_length=right_queue_length,
                right_weight=right_weight,
                right_pressure=right_pressure,
                right_expected_pressure=right_expected_pressure,
                right_cost_to_load=right_cost_to_load,
                right_claimants=right_claimants,
                right_loaded=right_loaded)
            left_qme["picked"] = pick_node == PICK_LEFT
            right_qme["picked"] = pick_node == PICK_RIGHT
            logger.log_event(
                "measure.queue.input",
                left_qme,
                adjust_ctx={
                    "executor": None,
                    "task": None,
                    "graph": left_graph_id,
                    "graph_name": left_graph_name,
                    "node": left_node.get_id(),
                    "node_name": left_node.get_name(),
                })
            logger.log_event(
                "measure.queue.input",
                right_qme,
                adjust_ctx={
                    "executor": None,
                    "task": None,
                    "graph": right_graph_id,
                    "graph_name": right_graph_name,
                    "node": right_node.get_id(),
                    "node_name": right_node.get_name(),
                })
            return left_node if pick_node == PICK_LEFT else right_node

        for node in self.get_all_nodes():
            if node == current_node:
                continue
            candidate_node = process_node(candidate_node, node)
        if current_node is None:
            return (candidate_node, True)
        own_queue = self.get_queue(current_node.get_input_queue())

        def own_queue_length() -> int:
            return own_queue.get_queue_length()

        def own_weight() -> float:
            return own_queue.total_weight()

        def own_pressure() -> float:
            return own_queue.total_backpressure()

        def own_expected_pressure() -> float:
            return own_queue.total_expected_pressure()

        def own_cost_to_load() -> float:
            return current_node.get_load_cost()

        def own_claimants() -> int:
            return own_queue.claimant_count()

        def own_loaded() -> int:
            return own_queue.get_listener_count()

        other_queue = self.get_queue(candidate_node.get_input_queue())

        def other_queue_length() -> int:
            return other_queue.get_queue_length()

        def other_weight() -> float:
            return other_queue.total_weight()

        def other_pressure() -> float:
            return other_queue.total_backpressure()

        def other_expected_pressure() -> float:
            return other_queue.total_expected_pressure()

        def other_cost_to_load() -> float:
            return candidate_node.get_load_cost()

        def other_claimants() -> int:
            return other_queue.claimant_count()

        def other_loaded() -> int:
            return other_queue.get_listener_count()

        if strategy.want_to_switch(
                own_queue_length=own_queue_length,
                own_weight=own_weight,
                own_pressure=own_pressure,
                own_expected_pressure=own_expected_pressure,
                own_cost_to_load=own_cost_to_load,
                own_claimants=own_claimants,
                own_loaded=own_loaded,
                other_queue_length=other_queue_length,
                other_weight=other_weight,
                other_pressure=other_pressure,
                other_expected_pressure=other_expected_pressure,
                other_cost_to_load=other_cost_to_load,
                other_claimants=other_claimants,
                other_loaded=other_loaded):
            return (candidate_node, candidate_node != current_node)
        return (current_node, False)

    def enqueue_task(
            self,
            ns: GNamespace,
            store: DataStore,
            original_input: TaskValueContainer) -> TaskId:
        """
        Enqueues a task to the overall input queue.

        Args:
            ns (GNamespace): The namespace.
            store (DataStore): The data store for storing the payload data.
            original_input (TaskValueContainer): The input data.

        Returns:
            TaskId: The new task id.
        """
        cpool = self.get_client_pool()
        task_id = cpool.create_task(ns, original_input)
        self._enqueue_task_id(cpool, store, task_id)
        return task_id

    def maybe_requeue_task_id(
            self,
            logger: EventStream,
            store: DataStore,
            task_id: TaskId,
            error_info: ErrorInfo) -> None:
        """
        Enqueue a previously created task again to the overall input queue.

        Args:
            logger (EventStream): The logger.
            store (DataStore): The data store for storing the payload data.
            task_id (TaskId): The existing task id.
            error_info (ErrorInfo): The error that triggered the requeue.

        Raises:
            AssertionError: If the task is already / still in a queue.
        """
        cpool = self.get_client_pool()
        cpool.set_error(task_id, error_info)
        new_retries = cpool.inc_retries(task_id)
        adj_ctx, retry_event = to_retry_event(error_info, new_retries)
        logger.log_event("error.task", retry_event, adjust_ctx=adj_ctx)
        if new_retries >= ClientPool.get_max_retries():
            cpool.set_duration(task_id)
            cpool.set_bulk_status([task_id], TASK_STATUS_ERROR)
        else:
            if cpool.get_status(task_id) == TASK_STATUS_WAIT:
                mqid = self.maybe_get_queue(task_id)
                if mqid is not None:
                    raise AssertionError(
                        f"{ctx_fmt()} {task_id} already waiting at {mqid}")
            cpool.clear_progress(task_id)
            self._enqueue_task_id(cpool, store, task_id)

    def handle_task_result(
            self,
            logger: EventStream,
            store: DataStore,
            task: ComputeTask) -> None:
        """
        Process the results of a computation on a task.

        Args:
            logger (EventStream): The logger.
            store (DataStore): The data store for storing the payload data.
            task (ComputeTask): The task that has the current result ready
                but has not committed and pushed the task to a new queue if
                any.
        """
        cpool = self.get_client_pool()
        data_id_type = store.data_id_type()
        task_id = task.get_task_id()
        ns = cpool.commit_task(
            task_id,
            task.get_data_out(),
            weight=task.get_weight_out(),
            byte_size=task.get_byte_size_out(),
            push_frame=task.get_push_frame())
        qid = task.get_next_queue_id()
        is_final = False
        while not is_final and qid.is_output_id():
            next_frame, frame_data = cpool.pop_frame(task_id, data_id_type)
            if next_frame is None:
                m_nname = None
                graph_id = self.get_entry_graph(ns)
                is_final = True
            else:
                m_nname, graph_id, qid = next_frame
            vmap = self.get_output_value_map(graph_id)
            ret_data = {
                QualifiedName(m_nname, vname): frame_data[frame_qual]
                for vname, frame_qual in vmap.items()
            }
            if is_final:
                final_vmap = {
                    qual.get_value_name(): qual
                    for qual, _ in ret_data.items()
                }
                output_format = self.get_output_format(graph_id)
                tvc_out = TaskValueContainer.extract_data(
                    store, output_format, ret_data, final_vmap)
                if tvc_out is not None:
                    cpool.set_final_output(task_id, tvc_out)
                    cpool.set_duration(task_id)
                    cpool.set_bulk_status([task_id], TASK_STATUS_READY)
                else:
                    self.maybe_requeue_task_id(
                        logger,
                        store,
                        task_id,
                        {
                            "ctx": get_ctx(),
                            "message": "result got purged from memory",
                            "code": "memory_purge",
                            "traceback": [],
                        })
            else:
                cpool.commit_task(
                    task_id,
                    ret_data,
                    weight=task.get_weight_out(),
                    byte_size=task.get_byte_size_out(),
                    push_frame=None)
        if not is_final:
            out_queue = self.get_queue(qid)
            out_queue.push_task_id(task_id)

    def get_task_status(self, task_id: TaskId) -> TaskStatus:
        """
        Retrieve the status of the given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            TaskStatus: The status of the task.
        """
        return self.get_client_pool().get_status(task_id)

    def _enqueue_task_id(
            self,
            cpool: ClientPool,
            store: DataStore,
            task_id: TaskId) -> None:
        """
        Raw enqueueing a task to the overall input.

        Args:
            cpool (ClientPool): The client pool.
            store (DataStore): The data store for storing the payload data.
            task_id (TaskId): The task id.
        """
        ns = cpool.get_namespace(task_id)
        assert ns is not None
        entry_graph_id = self.get_entry_graph(ns)
        input_format = self.get_input_format(entry_graph_id)
        cpool.init_data(store, task_id, input_format)
        node = self.get_input_node(entry_graph_id)
        qid = node.get_input_queue()
        cpool.set_bulk_status([task_id], TASK_STATUS_WAIT)
        self.push_task_id(qid, task_id)

    def get_compute_task(
            self,
            cpool: ClientPool,
            qid: QueueId,
            task_id: TaskId) -> ComputeTask:
        """
        Retrieve the current state of the given task.

        Args:
            cpool (ClientPool): The client pool.
            qid (QueueId): The queue id.
            task_id (TaskId): The task id.

        Returns:
            ComputeTask: The state of the task.
        """
        node = self.get_consumer_node(qid)
        return ComputeTask(cpool, task_id, node.get_value_map())

    def claim_compute_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[ComputeTask]:
        """
        Claim tasks for computing.

        Args:
            qid (QueueId): The queue.
            batch_size (int): The desired number of tasks.
            executor_id (ExecutorId): The executor that claims the tasks.

        Returns:
            list[ComputeTask]: The list of claimed tasks as states.
        """
        cpool = self.get_client_pool()
        task_ids = cpool.set_bulk_status(
            self.claim_tasks(qid, batch_size, executor_id), TASK_STATUS_BUSY)
        return [
            self.get_compute_task(cpool, qid, task_id)
            for task_id in task_ids
        ]

    def get_unclaimed_compute_tasks(self, qid: QueueId) -> list[ComputeTask]:
        """
        Get tasks in the queue that have not been claimed yet.

        Args:
            qid (QueueId): The queue id.

        Returns:
            list[ComputeTask]: The list of unclaimed tasks as states.
        """
        cpool = self.get_client_pool()
        return [
            self.get_compute_task(cpool, qid, task_id)
            for task_id in self.get_unclaimed_tasks(qid)
        ]

    def get_task_weight(self, task_id: TaskId) -> float:
        """
        Compute the weight / pressure of a given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            float: The weight of the task.
        """
        strategy = self.get_queue_strategy()
        return strategy.compute_weight(self.get_client_pool(), task_id)

    def push_task_id(self, qid: QueueId, task_id: TaskId) -> None:
        """
        Push a task to a queue.

        Args:
            qid (QueueId): The queue id.
            task_id (TaskId): The task id.
        """
        raise NotImplementedError()

    def get_unclaimed_tasks(self, qid: QueueId) -> list[TaskId]:
        """
        Retrieve unclaimed tasks in a queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            list[TaskId]: A list of unclaimed tasks as task ids.
        """
        raise NotImplementedError()

    def claim_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[TaskId]:
        """
        Claim tasks for execution.

        Args:
            qid (QueueId): The queue id.
            batch_size (int): The desired number of tasks to claim.
            executor_id (ExecutorId): The executor that claims the tasks.

        Returns:
            list[TaskId]: A list of claimed tasks as task ids.
        """
        raise NotImplementedError()

    def claimant_count(self, qid: QueueId) -> int:
        """
        Retrieves the number of executors having claims on the given queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            int: The number of executors having claims on the queue.
        """
        raise NotImplementedError()

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        """
        Unclaim tasks of an executor and make them available in the queue
        again.

        Args:
            qid (QueueId): The queue id.
            executor_id (ExecutorId): The executor id.

        Returns:
            list[TaskId]: The list of tasks that were associated with the
                executor as task ids.
        """
        raise NotImplementedError()

    def expect_task_weight(
            self,
            weight: float,
            byte_size: int,
            qid: QueueId,
            executor_id: ExecutorId) -> None:
        """
        Estimate the weight and payload size for the given queue by the given
        executor.

        Args:
            weight (float): The estimated weight that will be pushed to the
                queue.
            byte_size (int): The estimated amount of bytes that will be pushed
                to the queue.
            qid (QueueId): The affected queue.
            executor_id (ExecutorId): The executor that will push.
        """
        raise NotImplementedError()

    def clear_expected_task_weight(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        """
        Clear previously expected task weights and payload sizes for an
        executor.

        Args:
            qid (QueueId): The affected queue.
            executor_id (ExecutorId): The executor.
        """
        raise NotImplementedError()

    def get_expected_new_task_weight(self, qid: QueueId) -> float:
        """
        The sum of all currently expected new task weights for the given queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            float: The total expected weight on that queue.
        """
        raise NotImplementedError()

    def get_expected_byte_size(self, qid: QueueId) -> int:
        """
        The sum of all currently expected new payload sizes for the given
        queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            int: The total expected payload size on that queue.
        """
        raise NotImplementedError()

    def get_queue_length(self, qid: QueueId) -> int:
        """
        Retrieves the current length of the given queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            int: The current length of the queue.
        """
        raise NotImplementedError()

    def get_incoming_byte_size(self, qid: QueueId) -> int:
        """
        Compute the current incoming payload size on the given queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            int: The current payload size of the queue.
        """
        raise NotImplementedError()

    def clean_listeners(self, is_active: Callable[[ExecutorId], bool]) -> int:
        """
        Removes all listeners that are not active. If listener values are not
        parseable as executor (just a precaution) the value gets removed as
        well.

        Args:
            is_active (Callable[[ExecutorId], bool]): Returns True if the given
                executor is active and known.

        Returns:
            int: The number of cleaned up listeners.
        """
        raise NotImplementedError()

    def get_node_listeners(self, node: 'Node') -> int:
        """
        Countes how many listeners (executors) have loaded the given node.

        Args:
            node (Node): The node.

        Returns:
            int: The number of executors.
        """
        return self.get_queue_listeners(node.get_input_queue())

    def add_node_listener(self, node: 'Node', executor_id: ExecutorId) -> None:
        """
        Indicates the the executor has loaded the given node.

        Args:
            node (Node): The node.
            executor_id (ExecutorId): The executor.
        """
        self.add_queue_listener(node.get_input_queue(), executor_id)

    def remove_node_listener(
            self, node: 'Node', executor_id: ExecutorId) -> None:
        """
        Indicates the the executor has unloaded the given node.

        Args:
            node (Node): The node.
            executor_id (ExecutorId): The executor.
        """
        self.remove_queue_listener(
            qid=node.get_input_queue(), executor_id=executor_id)

    def get_queue_listeners(self, qid: QueueId) -> int:
        """
        Counts how many executors have loaded the node associated with this
        queue.

        Args:
            qid (QueueId): The queue id.

        Returns:
            int: The number of executors that have loaded the node associated
                with this queue.
        """
        raise NotImplementedError()

    def add_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        """
        Adds a listener (executor) that has loaded the node associated with
        this queue.

        Args:
            qid (QueueId): The queue id.
            executor_id (ExecutorId): The executor.
        """
        raise NotImplementedError()

    def remove_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        """
        Removes a listener (executor) that has loaded the node associated with
        this queue.

        Args:
            qid (QueueId): The queue id.
            executor_id (ExecutorId): The executor.
        """
        raise NotImplementedError()

    def maybe_get_queue(self, task_id: TaskId) -> QueueId | None:
        """
        Retrieve the queue the given task is in if any.

        Args:
            task_id (TaskId): The task id.

        Returns:
            QueueId | None: The queue the task is in or None if the task is not
                in a queue.
        """
        raise NotImplementedError()
