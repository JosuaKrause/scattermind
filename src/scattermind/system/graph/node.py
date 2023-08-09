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


# FIXME redis
class Node:
    def __init__(self, kind: str, graph: 'Graph', node_id: NodeId) -> None:
        self._kind = kind
        self._graph = graph
        self._node_id = node_id
        self._loads: set[ExecutorId] = set()
        self._load_lock = threading.RLock()

    def get_kind(self) -> str:
        return self._kind

    def get_id(self) -> NodeId:
        return self._node_id

    def get_name(self) -> NName:
        return self._graph.get_node_name(self._node_id)

    def get_graph(self) -> GraphId:
        return self._graph.get_graph_of(self.get_id())

    def get_context_info(self, queue_pool: QueuePool) -> ContextInfo:
        graph_id = self.get_graph()
        return {
            "node": self.get_id(),
            "node_name": self.get_name(),
            "graph": graph_id,
            "graph_name": queue_pool.get_graph_name(graph_id),
        }

    def get_node_arguments(self) -> NodeArgs:
        return self._graph.get_node_arguments(self._node_id)

    def get_arg(self, name: str) -> NodeArg:
        return self._graph.get_node_arguments(self._node_id)[name]

    def get_value_map(self) -> ValueMap:
        return self._graph.get_value_map(self._node_id)

    def get_input_queue(self) -> QueueId:
        return self._graph.get_input_queue(self._node_id)

    def get_output_queues(self) -> list[QueueId]:
        return self._graph.get_output_queues(self._node_id)

    def get_output_queue(self, qname: QName) -> QueueId:
        return self._graph.get_output_queue(self._node_id, qname)

    def expected_meta(
            self, state: ComputeState) -> dict[QueueId, tuple[float, int]]:
        return {
            self.get_output_queue(QName(qname)): meta
            for qname, meta in self.expected_output_meta(state).items()
        }

    def get_input_data_format(self) -> DataFormat:
        return DataFormat({
            key: DataInfo(dtype_name, dims)
            for key, (dtype_name, dims) in self.get_input_format().items()
        })

    def get_output_data_format(self, qname: QName) -> DataFormat:
        out_format = self.get_output_format()
        return DataFormat({
            key: DataInfo(dtype_name, dims)
            for key, (dtype_name, dims) in out_format[qname.get()].items()
        })

    def get_outputs(self) -> set[str]:
        return set(self.get_output_format().keys())

    def load(self, executor_id: ExecutorId, roa: ReadonlyAccess) -> None:
        with self._load_lock:
            assert executor_id not in self._loads
            if not self._loads:
                self.do_load(roa)
            self._loads.add(executor_id)

    def unload(self, executor_id: ExecutorId) -> None:
        with self._load_lock:
            self._loads.remove(executor_id)  # NOTE: raises KeyError if invalid
            if not self._loads:
                self.do_unload()

    def is_pure(self, queue_pool: QueuePool) -> bool:
        return self.do_is_pure(self._graph, queue_pool)

    def do_is_pure(self, graph: 'Graph', queue_pool: QueuePool) -> bool:
        raise NotImplementedError()

    def get_input_format(self) -> DataFormatJSON:
        raise NotImplementedError()

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        raise NotImplementedError()

    def get_weight(self) -> float:
        raise NotImplementedError()

    def get_load_cost(self) -> float:
        raise NotImplementedError()

    def do_load(self, roa: ReadonlyAccess) -> None:
        raise NotImplementedError()

    def do_unload(self) -> None:
        raise NotImplementedError()

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        # FIXME maybe don't pass the state here
        raise NotImplementedError()

    def execute_tasks(self, state: ComputeState) -> None:
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
