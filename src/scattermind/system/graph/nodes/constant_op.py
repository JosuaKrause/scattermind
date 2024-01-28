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
"""Perform an element-wise binary operation with a constant operand on a
tensor."""
from collections.abc import Callable
from typing import cast, get_args, Literal

import torch

from scattermind.system.base import NodeId
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

    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
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
