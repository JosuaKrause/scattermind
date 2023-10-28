from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool
from scattermind.system.queue.strategy.strategy import QueueStrategy


class SimpleQueueStrategy(  # pylint: disable=too-few-public-methods
        QueueStrategy):
    def compute_weight(
            self,
            cpool: ClientPool,
            task_id: TaskId) -> float:
        return cpool.get_weight(task_id) * (cpool.get_retries(task_id) + 1)
