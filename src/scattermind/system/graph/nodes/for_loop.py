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
"""Call a subgraph repeatedly until a counter reaches 0."""
import numpy as np

from scattermind.system.base import QueueId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.torch_util import create_tensor


class ForLoop(Node):
    """Call a subgraph repeatedly until a counter reaches 0."""
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
                val = create_tensor(np.array([init_value]), dtype="int")
            else:
                val[-1] -= 1
            out = state.create_single(val)
            if val[-1] > 0:
                state.push_call("inner", [task], {"stack": out}, gname)
            else:
                state.push_results("out", [task], {"stack": out})
