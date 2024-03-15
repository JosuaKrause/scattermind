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
"""Tests whether caching is active."""
from scattermind.system.base import GraphId, NodeId
from scattermind.system.client.client import ComputeTask
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.node import Node
from scattermind.system.info import DataFormatJSON
from scattermind.system.payload.values import ComputeState
from scattermind.system.queue.queue import QueuePool
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.redis_util import get_test_salt
from scattermind.system.torch_util import str_to_tensor, tensor_to_str


SEEN: set[str] = set()


def seen_prefix(node_id: NodeId, seed: str) -> str:
    """
    Creates a prefix unique to the current test case and node.

    Args:
        node_id (NodeId): The node.
        seed (str): A configuration dependent seed.

    Returns:
        str: The prefix for seen values.
    """
    salt = get_test_salt()
    return f"{salt}:{node_id.to_parseable()}:{seed}:"


class TestCache(Node):
    """Tests whether caching is active. The input remains unchanged when seen
    the first time. If a value is encountered twice the output is changed to
    include a postfix. This node maintains a global state. Use it only for test
    cases."""
    def do_is_pure(
            self,
            graph: Graph,
            queue_pool: QueuePool,
            pure_cache: dict[GraphId, bool]) -> bool:
        return True  # NOTE: we lie ;)

    def get_input_format(self) -> DataFormatJSON:
        return {
            "text": ("uint8", [None]),
        }

    def get_output_format(self) -> dict[str, DataFormatJSON]:
        return {
            "out": {
                "text": ("uint8", [None]),
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
        seed = self.get_arg("seed").get("str")
        prefix = seen_prefix(self.get_id(), seed)
        postfix = self.get_arg("postfix").get("str")
        inputs = state.get_values()
        outs: list[str] = []
        for text in inputs.get_data("text").iter_values():
            cur = tensor_to_str(text)
            cur_seen = f"{prefix}{cur}"
            print(f"test {cur}")
            if cur_seen in SEEN:
                cur = f"{cur}{postfix}"
                print(f"seen {cur} {cur_seen}")
            SEEN.add(cur_seen)
            outs.append(cur)

        for task, out in zip(inputs.get_current_tasks(), outs):
            state.push_results(
                "out",
                [task],
                {
                    "text": state.create_single(str_to_tensor(out)),
                })
