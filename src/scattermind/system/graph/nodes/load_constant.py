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
"""Load a constant tensor."""
import torch

from scattermind.system.base import NodeId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


class LoadConstant(Node):
    """Load a constant tensor."""
    def __init__(self, kind: str, graph: Graph, node_id: NodeId) -> None:
        super().__init__(kind, graph, node_id)
        self._constant: torch.Tensor | None = None

    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {}

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        data_info = self.get_arg("ret").get("info")
        return {
            "out": {
                "value": data_info.to_json(),
            },
        }

    def get_weight(self) -> float:
        return 1.0

    def get_load_cost(self) -> float:
        _, _, length = self.get_arg("data").get("data")
        return length

    def do_load(self, roa: ReadonlyAccess) -> None:
        data = self.get_arg("data").get("data")
        data_info = self.get_arg("ret").get("info")
        self._constant = roa.load_tensor(data, data_info)

    def do_unload(self) -> None:
        self._constant = None

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        tasks = list(state.get_inputs_tasks())
        return {
            "out": (len(tasks), ComputeTask.get_total_byte_size(tasks)),
        }

    def execute_tasks(self, state: ComputeState) -> None:
        assert self._constant is not None
        tasks = list(state.get_inputs_tasks())
        state.push_results(
            "out",
            tasks,
            {
                "value": state.create_uniform(torch.vstack([
                    torch.unsqueeze(self._constant, 0)
                    for _ in tasks
                ])),
            })
