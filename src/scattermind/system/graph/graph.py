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
"""This module defines the in-memory representation of the execution graph."""
from scattermind.system.base import GraphId, NodeId, QueueId
from scattermind.system.graph.args import NodeArg, NodeArgs, NodeArguments
from scattermind.system.graph.loader import load_node
from scattermind.system.graph.node import Node
from scattermind.system.names import GNamespace, NName, QName, ValueMap
from scattermind.system.queue.queue import QueuePool


class Graph:
    """
    The in-memory representation of the execution graph. The graph defines how
    nodes connect to each other, which queue ids are associated with which
    nodes, and specifies the mapping of node ids to nodes. The full execution
    graph is loaded at all times. Nodes, on the other hand, are loaded and
    unloaded on demand.
    """
    def __init__(self, ns: GNamespace) -> None:
        self._ns = ns
        self._nodes: dict[NodeId, Node] = {}
        self._names: dict[NodeId, NName] = {}
        self._args: dict[NodeId, NodeArgs] = {}
        self._graph: dict[NodeId, GraphId] = {}
        self._node_ids: dict[tuple[GraphId, NodeId], NodeId] = {}
        self._input_ids: dict[NodeId, QueueId] = {}
        self._output_ids: dict[NodeId, dict[QName, QueueId]] = {}
        self._vmaps: dict[NodeId, ValueMap] = {}

    def get_namespace(self) -> GNamespace:
        """
        Get the namespace of the graph.

        Returns:
            GNamespace: The namespace.
        """
        return self._ns

    def get_node_name(self, node_id: NodeId) -> NName:
        """
        Gets the node name associated with the given node id.

        Args:
            node_id (NodeId): The node id.

        Returns:
            NName: The name.
        """
        return self._names[node_id]

    def get_node_arguments(self, node_id: NodeId) -> NodeArgs:
        """
        Retrieves the arguments for the given node.

        Args:
            node_id (NodeId): The node id.

        Returns:
            NodeArgs: The arguments for the node.
        """
        return self._args[node_id]

    def add_node(
            self,
            queue_pool: QueuePool,
            graph_id: GraphId,
            *,
            kind: str,
            name: NName,
            node_id: NodeId | None,
            args: NodeArguments,
            fixed_input_queue_id: QueueId | None,
            vmap: ValueMap) -> None:
        """
        Add a node to the graph.

        Args:
            queue_pool (QueuePool): The queue pool.
            graph_id (GraphId): The graph id of the graph the node belongs to.
            kind (str): The node type. See
                ::py:function:`scattermind.system.graph.loader.load_node`.
            name (NName): The name of the node.
            node_id (NodeId | None): The (optional) node id. If None, the id
                gets inferred from the name of the node and the graph. The node
                id can be specified to allow node renames without changing all
                ids.
            args (NodeArguments): The arguments for constructing the node.
            fixed_input_queue_id (QueueId | None): The (optional) input queue
                id. If None, the id gets inferred from the name of the node and
                the graph. The input queue id can be specified to allow node
                renames without changing all ids.
            vmap (ValueMap): The value map that specifies the locations of the
                inputs of the node.

        Raises:
            ValueError: If there is a problem adding the node.
        """
        gname = queue_pool.get_graph_name(graph_id)
        ns = gname.get_namespace()
        if ns != self._ns:
            raise ValueError(f"mismatching namespaces: {ns} != {self._ns}")
        if node_id is None:
            node_id = NodeId.create(gname, name)
        if node_id in self._names:
            raise ValueError(f"node already added {node_id} {name}")
        node_key = (graph_id, node_id)
        if node_key in self._node_ids:
            raise ValueError(f"node already added {node_id} {name}")
        if node_id in self._graph:
            raise ValueError(
                f"ambiguous node id {node_id} between graphs "
                f"{graph_id} and {self._graph[node_id]}")
        self._graph[node_id] = graph_id
        node = load_node(self, kind, node_id)
        self._nodes[node_id] = node
        self._names[node_id] = name
        self._args[node_id] = NodeArg.from_node_arguments(ns, args)
        self._vmaps[node_id] = vmap
        self._node_ids[node_key] = node_id
        if fixed_input_queue_id is None:
            fixed_input_queue_id = QueueId.create(gname, name)
        self._input_ids[node_id] = fixed_input_queue_id
        queue_pool.register_node(node)

    def get_input_queue(self, node_id: NodeId) -> QueueId:
        """
        Retrieves the input queue of the given node.

        Args:
            node_id (NodeId): The node id.

        Returns:
            QueueId: The queue id.
        """
        return self._input_ids[node_id]

    def get_graph_of(self, node_id: NodeId) -> GraphId:
        """
        Retrieves the graph the given node is a part of.

        Args:
            node_id (NodeId): The node id.

        Returns:
            GraphId: The graph id.
        """
        return self._graph[node_id]

    def get_node(self, node_id: NodeId) -> Node:
        """
        Retrieves the node object for the given id.

        Args:
            node_id (NodeId): The node id.

        Returns:
            Node: The node object.
        """
        return self._nodes[node_id]

    def get_value_map(self, node_id: NodeId) -> ValueMap:
        """
        Retrieves the value map for the inputs of the given node.

        Args:
            node_id (NodeId): The node id.

        Returns:
            ValueMap: The value map for the given node.
        """
        return self._vmaps[node_id]

    def add_edge(
            self,
            *,
            output_node_id: NodeId,
            output_queue: QName,
            input_queue_id: QueueId) -> None:
        """
        Connects an output of a node to an input queue.

        Args:
            output_node_id (NodeId): The node id.
            output_queue (QName): The name of the output of the node.
            input_queue_id (QueueId): The input queue id to connect to.

        Raises:
            ValueError: If there is a problem adding the edge.
        """
        output_ids = self._output_ids.get(output_node_id)
        if output_ids is None:
            output_ids = {}
            self._output_ids[output_node_id] = output_ids
        if output_queue in output_ids:
            raise ValueError(
                f"output edge already added {output_node_id} {output_queue}")
        output_ids[output_queue] = input_queue_id

    def get_output_queues(self, node_id: NodeId) -> list[QueueId]:
        """
        Retrieves all output queues of a given node.

        Args:
            node_id (NodeId): The node id.

        Returns:
            list[QueueId]: The list of output queue ids.
        """
        return [
            self.get_output_queue(node_id, QName(qname))
            for qname in self.get_node(node_id).get_outputs()
        ]

    def get_output_queue(self, node_id: NodeId, qname: QName) -> QueueId:
        """
        Retrieves the output queue for a given node and output name.

        Args:
            node_id (NodeId): The node id.
            qname (QName): The output name.

        Returns:
            QueueId: The queue id.
        """
        return self._output_ids[node_id][qname]

    def is_pure(self, queue_pool: QueuePool, graph_id: GraphId) -> bool:
        """
        Whether the given graph is pure, that is, the output of the graph
        depends only on the input of the graph and the settings of the nodes.
        Only pure graphs can cache inputs.

        Args:
            queue_pool (QueuePool): The queue pool.
            graph_id (GraphId): The graph id.

        Returns:
            bool: A graph is pure if all its nodes are pure.
        """
        for node in self.traverse_graph(queue_pool, graph_id):
            if not node.is_pure(queue_pool):
                return False
        return True

    def traverse_graph(
            self, queue_pool: QueuePool, graph_id: GraphId) -> list[Node]:
        """
        Fully traverse an execution graph. Each reachable node of the graph
        gets returned exactly once.

        Args:
            queue_pool (QueuePool): The queue pool.
            graph_id (GraphId): The graph id.

        Returns:
            list[Node]: The list of reachable nodes.
        """
        node = queue_pool.get_input_node(graph_id)
        res: list[Node] = []
        self.collect_nodes(queue_pool, node, res)
        return res

    def collect_nodes(
            self,
            queue_pool: QueuePool,
            start: Node,
            result: list[Node]) -> None:
        """
        Collect all nodes reachable from the given start node. Each reachable
        node gets returned exactly once.

        Args:
            queue_pool (QueuePool): The queue pool.
            start (Node): The start node.
            result (list[Node]): Reachable nodes get added to this list.
        """
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            result.append(cur)
            for qid in cur.get_output_queues():
                node = queue_pool.get_queue(qid).get_consumer_node()
                if node not in seen:
                    stack.append(node)
                    seen.add(node)
