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
