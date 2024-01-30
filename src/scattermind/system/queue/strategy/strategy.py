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
from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool


class NodeStrategy:
    """A node strategy defines which node to load next. It scores nodes by
    input size and pressure and costs to load. The weight might not be enough
    to change the current active node in which case a load does not happen."""
    # FIXME break down pressure into components
    def own_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        """
        Computes the score of the currently active node. Higher scores are
        better.

        Args:
            queue_length (int): The length of the input queue.
            pressure (float): The pressure of the input queue.
            expected_pressure (float): The upcoming (expected new) pressure
                that potentially will be added to the input queue.
            cost_to_load (float): The cost to load the node.

        Returns:
            float: The score of the node. Higher values are better.
        """
        raise NotImplementedError()

    def other_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        """
        Computes the score of other nodes. Higher scores are better.

        Args:
            queue_length (int): The length of the input queue.
            pressure (float): The pressure of the input queue.
            expected_pressure (float): The upcoming (expected new) pressure
                that potentially will be added to the input queue.
            cost_to_load (float): The cost to load the node.

        Returns:
            float: The score of the node. Higher values are better. Only the
                node with the highest score is compared to the currently
                active node.
        """
        raise NotImplementedError()

    def want_to_switch(self, own_score: float, other_score: float) -> bool:
        """
        Whether the other score warrants a switch of nodes. Higher scores are
        better. Ideally, the better score should win.

        Args:
            own_score (float): The score of the currently active node.
            other_score (float): The score of the new candidate node.

        Returns:
            bool: True, if computation should switch to the candidate node.
                This should be the case if the other score is higher than the
                own score but implementations are free to choose when to
                trigger a switch.
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
