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
"""A simple queue strategy."""
from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool
from scattermind.system.queue.strategy.strategy import QueueStrategy


class SimpleQueueStrategy(  # pylint: disable=too-few-public-methods
        QueueStrategy):
    """The simple queue strategy directly uses the raw weight of the task but
    adjusting it using the number of retries."""
    def compute_weight(
            self,
            cpool: ClientPool,
            task_id: TaskId) -> float:
        return cpool.get_weight(task_id) * (cpool.get_retries(task_id) + 1)
