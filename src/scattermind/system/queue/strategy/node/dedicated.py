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
"""This strategy assumes that there is one executor available for every node.
This way executors can be sticky and don't have to change nodes very often."""
from collections.abc import Callable

from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    PICK_LEFT,
    PICK_RIGHT,
    PickNode,
)


class DedicatedNodeStrategy(NodeStrategy):
    """This strategy assumes that there is one executor available for every
    node. This way executors can be sticky and don't have to change nodes very
    often."""
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
        l_queue = left_queue_length()
        r_queue = right_queue_length()
        l_loaded = left_loaded()
        r_loaded = right_loaded()
        if l_loaded == 0 and l_queue > 0 and r_queue == 0:
            return PICK_LEFT
        if r_loaded == 0 and r_queue > 0 and l_queue == 0:
            return PICK_RIGHT
        if l_loaded + 1 < r_loaded:
            return PICK_LEFT
        if r_loaded + 1 < l_loaded:
            return PICK_RIGHT
        return PICK_LEFT if l_queue > r_queue else PICK_RIGHT

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
        own_loads = own_loaded()
        other_loads = other_loaded()
        own_len = own_queue_length()
        other_len = other_queue_length()
        if other_loads == 0 and other_len > 0 and own_len == 0:
            return True
        return own_loads > other_loads + 1
