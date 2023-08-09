import torch

from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


class MatSquare(Node):
    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        size = self.get_arg("size").get("int")
        return {
            "value": ("float", [size, size]),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        size = self.get_arg("size").get("int")
        return {
            "out": {
                "value": ("float", [size, size]),
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

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        tasks = list(state.get_inputs_tasks())
        return {
            "out": (len(tasks), ComputeTask.get_total_byte_size(tasks)),
        }

    def execute_tasks(self, state: ComputeState) -> None:
        inputs = state.get_values()
        val = inputs.get_data("value").get_uniform()
        state.push_results(
            "out",
            inputs.get_current_tasks(),
            {
                "value": state.create_uniform(torch.bmm(val, val)),
            })
