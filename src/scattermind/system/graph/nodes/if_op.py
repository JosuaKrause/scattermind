from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


class IfOp(Node):
    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {
            "condition": ("bool", [1]),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "then": {},
            "else": {},
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
            "then": (len(tasks), ComputeTask.get_total_byte_size(tasks)),
            "else": (0.0, 0),
        }

    def execute_tasks(self, state: ComputeState) -> None:
        values = state.get_values()
        thens = []
        elses = []
        for task, data in values.iter_values():
            if bool(data["condition"].item()):
                thens.append(task)
            else:
                elses.append(task)
        state.push_results("then", thens, {})
        state.push_results("else", elses, {})
