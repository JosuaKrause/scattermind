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
