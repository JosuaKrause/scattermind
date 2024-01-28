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
"""Element-wise binary operation on two tensors of the same shape."""
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
from scattermind.system.torch_util import same_mask


OpName = Literal[
    "add",
    "mul",
]
"""The operations supported by the `bin_op` node."""
ALL_OPS: set[OpName] = set(get_args(OpName))
"""All operations supported by the `bin_op` node."""


class BinOp(Node):
    """
    Element-wise binary operation on two tensors of the same shape.
    """
    def __init__(self, kind: str, graph: Graph, node_id: NodeId) -> None:
        super().__init__(kind, graph, node_id)
        self._op: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def do_is_pure(self, graph: Graph, queue_pool: QueuePool) -> bool:
        return True

    def get_input_format(self) -> DataFormatJSON:
        return {
            "left": self.get_arg("input").get("info").to_json(),
            "right": self.get_arg("input").get("info").to_json(),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "out": {
                "value": self.get_arg("input").get("info").to_json(),
            },
        }

    def get_weight(self) -> float:
        return 1.0

    def get_load_cost(self) -> float:
        return 1.0

    def do_load(self, roa: ReadonlyAccess) -> None:

        def add(
                val_left: torch.Tensor,
                val_right: torch.Tensor) -> torch.Tensor:
            return val_left + val_right

        def mul(
                val_left: torch.Tensor,
                val_right: torch.Tensor) -> torch.Tensor:
            return val_left * val_right

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
        tasks = list(state.get_inputs_tasks())
        return {
            "out": (len(tasks), ComputeTask.get_total_byte_size(tasks)),
        }

    def execute_tasks(self, state: ComputeState) -> None:
        assert self._op is not None
        inputs = state.get_values()
        left = inputs.get_data("left")
        right = inputs.get_data("right")
        if left.is_uniform():
            out = state.create_uniform(
                self._op(left.get_uniform(), right.get_uniform()))
        else:
            left_val, left_mask = left.get_masked()
            right_val, right_mask = right.get_masked()
            if not same_mask(left_mask, right_mask):
                raise ValueError("invalid inputs")
            out = state.create_masked(self._op(left_val, right_val), left_mask)
        state.push_results(
            "out",
            inputs.get_current_tasks(),
            {
                "value": out,
            })
