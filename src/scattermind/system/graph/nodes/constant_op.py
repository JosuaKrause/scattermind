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
"""Perform an element-wise binary operation with a constant operand on a
tensor."""
from collections.abc import Callable
from typing import cast, get_args, Literal

import torch

from scattermind.system.base import GraphId, NodeId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


OpName = Literal[
    "add",
    "mul",
]
"""The operations supported by the `constant_op` node."""
ALL_OPS: set[OpName] = set(get_args(OpName))
"""All operations supported by the `constant_op` node."""


class ConstantOp(Node):
    """Perform an element-wise binary operation with a constant operand on a
    tensor."""
    def __init__(self, kind: str, graph: Graph, node_id: NodeId) -> None:
        super().__init__(kind, graph, node_id)
        self._op: Callable[[torch.Tensor], torch.Tensor] | None = None

    def do_is_pure(
            self,
            graph: Graph,
            queue_pool: QueuePool,
            pure_cache: dict[GraphId, bool]) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        data_info = self.get_arg("input").get("info")
        return {
            "value": data_info.to_json(),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        data_info = self.get_arg("input").get("info")
        return {
            "out": {
                "value": data_info.to_json(),
            },
        }

    def get_weight(self) -> float:
        return 1.0

    def get_load_cost(self) -> float:
        return 1.0

    def do_load(self, roa: ReadonlyAccess) -> None:
        val = float(self.get_arg("const").get("float"))

        def add(val_in: torch.Tensor) -> torch.Tensor:
            return val_in + val

        def mul(val_in: torch.Tensor) -> torch.Tensor:
            return val_in * val

        op: OpName = cast(OpName, self.get_arg("op").get("str"))
        if op == "add":
            self._op = add
        elif op == "mul":
            self._op = mul
        else:
            raise ValueError(f"unknown op: {op} valid ops: {ALL_OPS}")

    def do_unload(self) -> None:
        self._op = None

    def expected_output_meta(
            self, state: ComputeState) -> dict[str, tuple[float, int]]:
        # FIXME maybe remove expected output meta?
        tasks = list(state.get_inputs_tasks())
        return {
            "out": (len(tasks), ComputeTask.get_total_byte_size(tasks)),
        }

    def execute_tasks(self, state: ComputeState) -> None:
        assert self._op is not None
        inputs = state.get_values()
        value = inputs.get_data("value")
        if value.is_uniform():
            out = state.create_uniform(self._op(value.get_uniform()))
        else:
            val, mask = value.get_masked()
            out = state.create_masked(self._op(val), mask)
        state.push_results(
            "out",
            inputs.get_current_tasks(),
            {
                "value": out,
            })
