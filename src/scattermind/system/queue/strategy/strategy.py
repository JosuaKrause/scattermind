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
from typing import Literal

from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool


PickNode = Literal["left", "right"]
"""Which node to pick."""
PICK_LEFT: PickNode = "left"
"""Picked the left node."""
PICK_RIGHT: PickNode = "right"
"""Picked the right node."""


class NodeStrategy:
    """A node strategy defines which node to load next. It compares nodes by
    input size and pressure and costs to load. The weight might not be enough
    to change the current active node in which case a load does not happen."""
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
        """
        Compares two nodes and decides which node to pick.

        Args:
            left_queue_length (Callable[[], int]): The length of the left input
                queue.
            left_weight (Callable[[], float]): The weight of the left input
                queue.
            left_pressure (Callable[[], float]): The pressure of the left input
                queue.
            left_expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                left input queue.
            left_cost_to_load (Callable[[], float]): The cost to load the left
                node.
            left_claimants (Callable[[], int]): The number of executors
                currently laying claim to the left node.
            left_loaded (Callable[[], int]): The number of executors that
                loaded the left node.
            right_queue_length (Callable[[], int]): The length of the right
                input queue.
            right_weight (Callable[[], float]): The weight of the right input
                queue.
            right_pressure (Callable[[], float]): The pressure of the right
                input queue.
            right_expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                right input queue.
            right_cost_to_load (Callable[[], float]): The cost to load the
                right node.
            right_claimants (Callable[[], int]): The number of executors
                currently laying claim to the right node.
            right_loaded (Callable[[], int]): The number of executors that
                loaded the right node.

        Returns:
            PickNode: Which node to pick out of the two.
        """
        raise NotImplementedError()

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
        """
        Whether the other node info warrants a switch of nodes.

        Args:
            own_queue_length (Callable[[], int]): The length of the own input
                queue.
            own_weight (Callable[[], float]): The weight of the own input
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
            own_loaded (Callable[[], int]): The number of executors that
                loaded the own node.
            other_queue_length (Callable[[], int]): The length of the other
                input queue.
            other_weight (Callable[[], float]): The weight of the other input
                queue.
            other_pressure (Callable[[], float]): The pressure of the other
                input queue.
            other_expected_pressure (Callable[[], float]): The upcoming
                (expected new) pressure that potentially will be added to the
                other input queue.
            other_cost_to_load (Callable[[], float]): The cost to load the
                other node.
            other_claimants (Callable[[], int]): The number of executors
                currently laying claim to the other node.
            other_loaded (Callable[[], int]): The number of executors that
                loaded the other node.

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
