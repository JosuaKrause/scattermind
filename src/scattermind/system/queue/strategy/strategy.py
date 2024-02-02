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
"""Defines the strategy interfaces."""
from collections.abc import Callable

from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool


class NodeStrategy:
    """A node strategy defines which node to load next. It scores nodes by
    input size and pressure and costs to load. The weight might not be enough
    to change the current active node in which case a load does not happen."""
    # FIXME break down pressure into components
    def other_score(
            self,
            *,
            queue_length: Callable[[], int],
            pressure: Callable[[], float],
            expected_pressure: Callable[[], float],
            cost_to_load: Callable[[], float],
            claimants: Callable[[], int]) -> float:
        """
        Computes the score of other nodes. Higher scores are better.

        Args:
            queue_length (Callable[[], int]): The length of the input queue.
            pressure (Callable[[], float]): The pressure of the input queue.
            expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                input queue.
            cost_to_load (Callable[[], float]): The cost to load the node.
            claimants (Callable[[], int]): The number of executors currently
                laying claim to the node.

        Returns:
            float: The score of the node. Higher values are better. Only the
                node with the highest score is compared to the currently
                active node.
        """
        raise NotImplementedError()

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
        """
        Whether the other score warrants a switch of nodes. Higher scores are
        better. Ideally, the better score should win.

        Args:
            own_queue_length (Callable[[], int]): The length of the own input
                queue.
            own_pressure (Callable[[], float]): The pressure of the own input
                queue.
            own_expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                own input queue.
            own_cost_to_load (Callable[[], float]): The cost to load the own
                node.
            own_claimants (Callable[[], int]): The number of executors
                currently laying claim to the own node.
            other_queue_length (Callable[[], int]): The length of the other
                input queue.
            other_pressure (Callable[[], float]): The pressure of the other
                input queue.
            other_expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                other input queue.
            other_cost_to_load (Callable[[], float]): The cost to load the
                other node.
            other_claimants (Callable[[], int]): The number of executors
                currently laying claim to the other node.

        Returns:
            bool: True, if computation should switch to the candidate node.
        """
        raise NotImplementedError()


class QueueStrategy:  # pylint: disable=too-few-public-methods
    """A queue strategy assigns weights to tasks in a queue. The top
    `batch_size` tasks are then loaded and processed."""
    def compute_weight(
            self,
            cpool: ClientPool,
            task_id: TaskId) -> float:
        """
        Computes the weight of a task.

        Args:
            cpool (ClientPool): The client pool.
            task_id (TaskId): The task id.

        Returns:
            float: The weight for picking tasks. Tasks are chosen from the
                highest weight down.
        """
        raise NotImplementedError()
