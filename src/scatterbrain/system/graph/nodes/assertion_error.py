
from scatterbrain.system.graph.graph import Graph
from scatterbrain.system.graph.node import Node
from scatterbrain.system.info import DataFormatJSON
from scatterbrain.system.payload.values import ComputeState
from scatterbrain.system.queue.queue import QueuePool
from scatterbrain.system.readonly.access import ReadonlyAccess


class AssertionErrorNode(Node):
    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {}

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {}

    def get_weight(self) -> float:
        return 1.0

    def get_load_cost(self) -> float:
        return 1.0

    def do_load(self, roa: ReadonlyAccess) -> None:
        pass

    def do_unload(self) -> None:
        pass

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        return {}

    def execute_tasks(self, state: ComputeState) -> None:
        msg = self.get_arg("msg").get("str")
        raise ValueError(msg)
