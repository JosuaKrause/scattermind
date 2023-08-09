import numpy as np

from scatterbrain.system.base import QueueId
from scatterbrain.system.client.client import ComputeTask
from scatterbrain.system.graph.graph import Graph
from scatterbrain.system.graph.node import Node
from scatterbrain.system.info import DataFormatJSON
from scatterbrain.system.payload.values import ComputeState
from scatterbrain.system.queue.queue import QueuePool
from scatterbrain.system.readonly.access import ReadonlyAccess
from scatterbrain.system.torch_util import create_tensor


class ForLoop(Node):
    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {
            "stack": ("int", [None]),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "inner": {
                "stack": ("int", [None]),
            },
            "out": {
                "stack": ("int", [None]),
            },
        }

    def get_weight(self) -> float:
        return 1.0

    def get_load_cost(self) -> float:
        return 1.0

    def do_load(self, roa: ReadonlyAccess) -> None:
        pass

    def do_unload(self) -> None:
        pass

    def expected_meta(
            self, state: ComputeState) -> dict[QueueId, tuple[float, int]]:
        gname = self.get_arg("graph").get("graph")
        tasks = list(state.get_inputs_tasks())
        return {
            state.get_graph_input_queue(gname):
                (len(tasks), ComputeTask.get_total_byte_size(tasks)),
        }

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        raise AssertionError("should not be called")

    def execute_tasks(self, state: ComputeState) -> None:
        gname = self.get_arg("graph").get("graph")
        init_value = self.get_arg("init").get("int")
        for task, value in state.get_values().iter_values():
            val = value["stack"]
            if not val.shape[0]:
                val = create_tensor(np.array([init_value]), "int")
            else:
                val[-1] -= 1
            out = state.create_single(val)
            if val[-1] > 0:
                state.push_call("inner", [task], {"stack": out}, gname)
            else:
                state.push_results("out", [task], {"stack": out})
