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
"""Call a subgraph for execution and present its results as the node's."""
from scattermind.system.base import GraphId, QueueId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess


class Call(Node):
    """
    Call a subgraph for execution. The node arguments are the inputs to the
    subgraph and the output of the node is the output of the subgraph.
    """
    def do_is_pure(
            self,
            graph: Graph,
            queue_pool: QueuePool,
            pure_cache: dict[GraphId, bool]) -> bool:
        return graph.is_pure(
            queue_pool,
            graph.get_graph_of(self.get_id()),
            pure_cache=pure_cache)

    def get_input_format(self) -> DataFormatJSON:
        return self.get_arg("args").get("format")

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "out": self.get_arg("ret").get("format"),
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
        values = state.get_values()
        out = {
            vname: state.create(values.get_data(vname))
            for vname in values.get_value_names()
        }
        print(f"{ctx_fmt()} call out: {out}")
        state.push_call("out", values.get_current_tasks(), out, gname)
