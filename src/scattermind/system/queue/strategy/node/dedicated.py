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

from scattermind.system.queue.strategy.strategy import NodeStrategy


class DedicatedNodeStrategy(NodeStrategy):
    """This strategy assumes that there is one executor available for every
    node. This way executors can be sticky and don't have to change nodes very
    often."""
    def other_score(
            self,
            *,
            queue_length: Callable[[], int],
            pressure: Callable[[], float],
            expected_pressure: Callable[[], float],
            cost_to_load: Callable[[], float],
            claimants: Callable[[], int]) -> float:
        return queue_length() / (claimants() + 1)

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
        if own_claimants() >= other_claimants() + 2:
            return (
                other_queue_length() > 0
                or other_pressure() > 0
                or other_expected_pressure() > 0)
        return False
