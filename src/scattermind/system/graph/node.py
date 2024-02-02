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
"""A node in the execution graph."""
import threading
from typing import TYPE_CHECKING

from scattermind.system.base import ExecutorId, GraphId, NodeId, QueueId
from scattermind.system.graph.args import NodeArg, NodeArgs
from scattermind.system.info import DataFormat, DataFormatJSON, DataInfo
from scattermind.system.logger.context import ContextInfo
from scattermind.system.names import NName, QName, ValueMap
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.util import shorthand_if_mod


if TYPE_CHECKING:
    from scattermind.system.graph.graph import Graph


INTERNAL_NODE_PREFIX = "scattermind.system.graph.nodes"
"""Python module prefix for built-in nodes."""


# FIXME redis
class Node:
    """
    A node of the execution graph. Overwrite this class to implement new
    functionality.
    """
    def __init__(self, kind: str, graph: 'Graph', node_id: NodeId) -> None:
        """
        Creates a node. If overwriting this constructor keep the signature the
        same. Use node arguments to provide node customization.

        Args:
            kind (str): The kind of the node.
            graph (Graph): The graph of the node.
            node_id (NodeId): The node id.
        """
        self._kind = kind
        self._graph = graph
        self._node_id = node_id
        self._loads: set[ExecutorId] = set()
        self._load_lock = threading.RLock()

    def get_kind(self) -> str:
        """
        The kind of the node.

        Returns:
            str: The kind.
        """
        return self._kind

    def get_id(self) -> NodeId:
        """
        The id of the node.

        Returns:
            NodeId: The id.
        """
        return self._node_id

    def get_name(self) -> NName:
        """
        Retrieves the name of the node.

        Returns:
            NName: The name.
        """
        return self._graph.get_node_name(self._node_id)

    def get_graph(self) -> GraphId:
        """
        Retrieves the graph the node belongs to.

        Returns:
            GraphId: The graph id.
        """
        return self._graph.get_graph_of(self.get_id())

    def get_context_info(self, queue_pool: QueuePool) -> ContextInfo:
        """
        Provide information about the node.

        Args:
            queue_pool (QueuePool): The queue pool.

        Returns:
            ContextInfo: Returns information about the id, name, graph, and
                graph name associated with the node.
        """
        graph_id = self.get_graph()
        return {
            "node": self.get_id(),
            "node_name": self.get_name(),
            "graph": graph_id,
            "graph_name": queue_pool.get_graph_name(graph_id),
        }

    def get_node_arguments(self) -> NodeArgs:
        """
        Retrieves the arguments to the node.

        Returns:
            NodeArgs: The node argument values.
        """
        return self._graph.get_node_arguments(self._node_id)

    def get_arg(self, name: str) -> NodeArg:
        """
        Get an argument by name.

        Args:
            name (str): The name.

        Returns:
            NodeArg: The node argument value.
        """
        return self._graph.get_node_arguments(self._node_id)[name]

    def get_value_map(self) -> ValueMap:
        """
        The value map of the node to determine the locations of its inputs.

        Returns:
            ValueMap: The value map.
        """
        return self._graph.get_value_map(self._node_id)

    def get_input_queue(self) -> QueueId:
        """
        The input queue of the node.

        Returns:
            QueueId: The queue id.
        """
        return self._graph.get_input_queue(self._node_id)

    def get_output_queues(self) -> list[QueueId]:
        """
        Retrieves the output queues of the node.

        Returns:
            list[QueueId]: The queue ids.
        """
        return self._graph.get_output_queues(self._node_id)

    def get_output_queue(self, qname: QName) -> QueueId:
        """
        Retrieve the output queue for a given name.

        Args:
            qname (QName): The output name.

        Returns:
            QueueId: The queue id.
        """
        return self._graph.get_output_queue(self._node_id, qname)

    def expected_meta(
            self, state: ComputeState) -> dict[QueueId, tuple[float, int]]:
        """
        Retrieve the expected distribution of tasks for each output queue.
        The meta data consists of the expected weight and byte size for the
        output queues.

        Args:
            state (ComputeState): The compute state.

        Returns:
            dict[QueueId, tuple[float, int]]: Dictionary mapping the output
                queues to the expected meta data.
        """
        return {
            self.get_output_queue(QName(qname)): meta
            for qname, meta in self.expected_output_meta(state).items()
        }

    def get_input_data_format(self) -> DataFormat:
        """
        Get the format of the input data of the node.

        Returns:
            DataFormat: The expected data format.
        """
        return DataFormat({
            key: DataInfo(dtype_name, dims)
            for key, (dtype_name, dims) in self.get_input_format().items()
        })

    def get_output_data_format(self, qname: QName) -> DataFormat:
        """
        Retrieve the format of the output data for the given output.

        Args:
            qname (QName): The output name.

        Returns:
            DataFormat: The expected data format.
        """
        out_format = self.get_output_format()
        return DataFormat({
            key: DataInfo(dtype_name, dims)
            for key, (dtype_name, dims) in out_format[qname.get()].items()
        })

    def get_outputs(self) -> set[str]:
        """
        Get all valid output names.

        Returns:
            set[str]: A set of all outputs.
        """
        return set(self.get_output_format().keys())

    def load(self, executor_id: ExecutorId, roa: ReadonlyAccess) -> None:
        """
        Load the node for the given executor.

        Args:
            executor_id (ExecutorId): The executor loading the node.
            roa (ReadonlyAccess): The readonly data access.
        """
        with self._load_lock:
            assert executor_id not in self._loads
            if not self._loads:
                self.do_load(roa)
            self._loads.add(executor_id)

    def unload(self, executor_id: ExecutorId) -> None:
        """
        Unload the node and release resources.

        Args:
            executor_id (ExecutorId): The executor unloading the node.
        """
        with self._load_lock:
            self._loads.remove(executor_id)  # NOTE: raises KeyError if invalid
            if not self._loads:
                self.do_unload()

    def is_pure(self, queue_pool: QueuePool) -> bool:
        """
        Whether the computation of the node is "pure", i.e., whether the result
        of the node is deterministic and *only* depends on its inputs (and
        static settings).

        See also :py:method::`
        scattermind.system.payload.data.DataStore#is_content_addressable`.

        Args:
            queue_pool (QueuePool): The queue pool.

        Returns:
            bool: True if the node returns the same result for same inputs
                given the settings are the same.
        """
        # FIXME actually implement a way of caching node outputs
        return self.do_is_pure(self._graph, queue_pool)

    def do_is_pure(self, graph: 'Graph', queue_pool: QueuePool) -> bool:
        """
        Whether the computation of the node is "pure", i.e., whether the result
        of the node is deterministic and *only* depends on its inputs (and
        static settings).

        See also :py:method::`
        scattermind.system.payload.data.DataStore#is_content_addressable`.

        Args:
            graph (Graph): The node's graph.
            queue_pool (QueuePool): The queue pool.

        Returns:
            bool: True if the node returns the same result for same inputs
                given the settings are the same.
        """
        raise NotImplementedError()

    def get_input_format(self) -> DataFormatJSON:
        """
        The expected input format of the node.

        Returns:
            DataFormatJSON: The input format.
        """
        raise NotImplementedError()

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        """
        The expected output formats of the node.

        Returns:
            dict[str, DataFormatJSON]: Dictionary mapping each output to its
                expected format.
        """
        raise NotImplementedError()

    def get_weight(self) -> float:
        """
        The added weight for each task.

        Returns:
            float: The weight added to a task that is processed by this node.
        """
        raise NotImplementedError()

    def get_load_cost(self) -> float:
        """
        The estimated cost of loading the node.

        Returns:
            float: The estimated cost.
        """
        raise NotImplementedError()

    def do_load(self, roa: ReadonlyAccess) -> None:
        """
        Load all resources to make the node executable.

        Args:
            roa (ReadonlyAccess): The readonly data access.
        """
        raise NotImplementedError()

    def do_unload(self) -> None:
        """
        Unload the node and ensure that unneeded memory is freed.
        """
        raise NotImplementedError()

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        """
        Estimate the expected weights and byte sizes for each output.

        Args:
            state (ComputeState): The compute state.

        Returns:
            dict[str, tuple[float, int]]: The expected weights and byte sizes
                for each output.
        """
        # FIXME maybe don't pass the state here
        raise NotImplementedError()

    def execute_tasks(self, state: ComputeState) -> None:
        """
        Execute the node for the given data. Results can be pushed using the
        respective functions in the compute state object.

        Args:
            state (ComputeState): The compute state holding the data.
        """
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, Node):
            return False
        return self.get_id() == other.get_id()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.get_id())

    def __str__(self) -> str:
        return (
            f"{shorthand_if_mod(self.__class__, INTERNAL_NODE_PREFIX)}["
            f"{self._node_id.to_parseable()},"
            f"{self.get_name().get()}]")

    def __repr__(self) -> str:
        return self.__str__()
