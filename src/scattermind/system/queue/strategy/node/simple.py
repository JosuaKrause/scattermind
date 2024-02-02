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

from scattermind.system.queue.strategy.strategy import NodeStrategy


class SimpleNodeStrategy(NodeStrategy):
    """The simple node strategy tallies up the queue length and pressure and
    scales it down by cost to load."""
    def other_score(
            self,
            *,
            queue_length: Callable[[], int],
            pressure: Callable[[], float],
            expected_pressure: Callable[[], float],
            cost_to_load: Callable[[], float],
            claimants: Callable[[], int]) -> float:
        numerator = queue_length() + expected_pressure() + pressure()
        return numerator / cost_to_load()

    def want_to_switch(
            self,
            own_queue_length: Callable[[], int],
            own_pressure: Callable[[], float],
            own_expected_pressure: Callable[[], float],
            own_cost_to_load: Callable[[], float],
            own_claimants: Callable[[], int],
            other_queue_length: Callable[[], int],
            other_pressure: Callable[[], float],
            other_expected_pressure: Callable[[], float],
            other_cost_to_load: Callable[[], float],
            other_claimants: Callable[[], int]) -> bool:
        own_num = own_queue_length() + own_expected_pressure() + own_pressure()
        own_score = own_num / own_cost_to_load()
        other_num = (
            other_queue_length()
            + other_expected_pressure()
            + other_pressure())
        other_score = other_num / other_cost_to_load()
        return other_score > own_score
