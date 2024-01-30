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
from scattermind.system.queue.strategy.strategy import NodeStrategy


class SimpleNodeStrategy(NodeStrategy):
    """The simple node strategy tallies up the queue length and pressure and
    scales it down by cost to load."""
    def own_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        return (queue_length + expected_pressure + pressure) / cost_to_load

    def other_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        return (queue_length + expected_pressure + pressure) / cost_to_load

    def want_to_switch(self, own_score: float, other_score: float) -> bool:
        return other_score > own_score
