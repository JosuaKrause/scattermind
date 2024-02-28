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
"""A simple node strategy."""
from collections.abc import Callable

from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    PICK_LEFT,
    PICK_RIGHT,
    PickNode,
)


class SimpleNodeStrategy(NodeStrategy):
    """The simple node strategy."""
    def pick_node(
            self,
            *,
            left_queue_length: Callable[[], int],
            left_weight: Callable[[], float],
            left_pressure: Callable[[], float],
            left_expected_pressure: Callable[[], float],
            left_cost_to_load: Callable[[], float],
            left_claimants: Callable[[], int],
            left_loaded: Callable[[], int],
            right_queue_length: Callable[[], int],
            right_weight: Callable[[], float],
            right_pressure: Callable[[], float],
            right_expected_pressure: Callable[[], float],
            right_cost_to_load: Callable[[], float],
            right_claimants: Callable[[], int],
            right_loaded: Callable[[], int]) -> PickNode:
        left_numerator = (
            left_queue_length()
            + left_expected_pressure()
            + left_pressure())
        left_score = left_numerator / left_cost_to_load()
        right_numerator = (
            right_queue_length()
            + right_expected_pressure()
            + right_pressure())
        right_score = right_numerator / right_cost_to_load()
        return PICK_LEFT if left_score > right_score else PICK_RIGHT

    def want_to_switch(
            self,
            own_queue_length: Callable[[], int],
            own_weight: Callable[[], float],
            own_pressure: Callable[[], float],
            own_expected_pressure: Callable[[], float],
            own_cost_to_load: Callable[[], float],
            own_claimants: Callable[[], int],
            own_loaded: Callable[[], int],
            other_queue_length: Callable[[], int],
            other_weight: Callable[[], float],
            other_pressure: Callable[[], float],
            other_expected_pressure: Callable[[], float],
            other_cost_to_load: Callable[[], float],
            other_claimants: Callable[[], int],
            other_loaded: Callable[[], int]) -> bool:
        own_num = own_queue_length() + own_expected_pressure() + own_pressure()
        own_score = own_num / own_cost_to_load()
        other_num = (
            other_queue_length()
            + other_expected_pressure()
            + other_pressure())
        other_score = other_num / other_cost_to_load()
        return other_score > own_score
