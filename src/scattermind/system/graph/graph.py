from scattermind.system.base import GraphId, NodeId, QueueId
from scattermind.system.graph.args import NodeArg, NodeArgs, NodeArguments
from scattermind.system.graph.loader import load_node
from scattermind.system.graph.node import Node
from scattermind.system.names import NName, QName, ValueMap
from scattermind.system.queue.queue import QueuePool


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[NodeId, Node] = {}
        self._names: dict[NodeId, NName] = {}
        self._args: dict[NodeId, NodeArgs] = {}
        self._graph: dict[NodeId, GraphId] = {}
        self._node_ids: dict[tuple[GraphId, NodeId], NodeId] = {}
        self._input_ids: dict[NodeId, QueueId] = {}
        self._output_ids: dict[NodeId, dict[QName, QueueId]] = {}
        self._vmaps: dict[NodeId, ValueMap] = {}

    def get_node_name(self, node_id: NodeId) -> NName:
        return self._names[node_id]

    def get_node_arguments(self, node_id: NodeId) -> NodeArgs:
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
        gname = queue_pool.get_graph_name(graph_id)
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
        self._args[node_id] = NodeArg.from_node_arguments(args)
        self._vmaps[node_id] = vmap
        self._node_ids[node_key] = node_id
        if fixed_input_queue_id is None:
            fixed_input_queue_id = QueueId.create(gname, name)
        self._input_ids[node_id] = fixed_input_queue_id
        queue_pool.register_node(node)

    def get_input_queue(self, node_id: NodeId) -> QueueId:
        return self._input_ids[node_id]

    def get_graph_of(self, node_id: NodeId) -> GraphId:
        return self._graph[node_id]

    def get_node(self, node_id: NodeId) -> Node:
        return self._nodes[node_id]

    def get_value_map(self, node_id: NodeId) -> ValueMap:
        return self._vmaps[node_id]

    def add_edge(
            self,
            *,
            output_node_id: NodeId,
            output_queue: QName,
            input_queue_id: QueueId) -> None:
        output_ids = self._output_ids.get(output_node_id)
        if output_ids is None:
            output_ids = {}
            self._output_ids[output_node_id] = output_ids
        if output_queue in output_ids:
            raise ValueError(
                f"output edge already added {output_node_id} {output_queue}")
        output_ids[output_queue] = input_queue_id

    def get_output_queues(self, node_id: NodeId) -> list[QueueId]:
        return [
            self.get_output_queue(node_id, QName(qname))
            for qname in self.get_node(node_id).get_outputs()
        ]

    def get_output_queue(self, node_id: NodeId, qname: QName) -> QueueId:
        return self._output_ids[node_id][qname]

    def is_pure(self, queue_pool: QueuePool, graph_id: GraphId) -> bool:
        for node in self.traverse_graph(queue_pool, graph_id):
            if not node.is_pure(queue_pool):
                return False
        return True

    def traverse_graph(
            self, queue_pool: QueuePool, graph_id: GraphId) -> list[Node]:
        node = queue_pool.get_input_node(graph_id)
        res: list[Node] = []
        self.collect_nodes(queue_pool, node, res)
        return res

    def collect_nodes(
            self,
            queue_pool: QueuePool,
            start: Node,
            result: list[Node]) -> None:
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            result.append(cur)
            for qid in cur.get_output_queues():
                node = queue_pool.get_queue(qid).get_consumer_node()
                if node not in seen:
                    stack.append(node)
                    seen.add(node)
