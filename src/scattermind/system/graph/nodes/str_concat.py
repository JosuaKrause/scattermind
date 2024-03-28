# Copyright (C) 2024 Josua Krause
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Concatenate two strings."""
from scattermind.system.base import GraphId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.torch_util import str_to_tensor, tensor_to_str


class StrConcat(Node):
    """Concatenate two strings."""
    def do_is_pure(
            self,
            graph: Graph,
            queue_pool: QueuePool,
            pure_cache: dict[GraphId, bool]) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {
            "left": ("uint8", [None]),
            "right": ("uint8", [None]),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "out": {
                "value": ("uint8", [None]),
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
        delimiter = self.get_arg("delimiter").get("str")
        inputs = state.get_values()
        lefts: list[str] = [
            tensor_to_str(left)
            for left in inputs.get_data("left").iter_values()
        ]
        rights: list[str] = [
            tensor_to_str(right)
            for right in inputs.get_data("right").iter_values()
        ]

        outs: list[str] = [
            f"{left}{delimiter}{right}"
            for left, right in zip(lefts, rights)
        ]
        for task, out in zip(inputs.get_current_tasks(), outs):
            state.push_results(
                "out",
                [task],
                {
                    "value": state.create_single(str_to_tensor(out)),
                })
