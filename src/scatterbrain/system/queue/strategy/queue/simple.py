from scatterbrain.system.base import TaskId
from scatterbrain.system.client.client import ClientPool
from scatterbrain.system.queue.strategy.strategy import QueueStrategy


class SimpleQueueStrategy(  # pylint: disable=too-few-public-methods
        QueueStrategy):
    def sort_queue(
            self,
            cpool: ClientPool,
            candidates: list[TaskId]) -> None:

        def compute_weight(task_id: TaskId) -> float:
            return cpool.get_weight(task_id) * (cpool.get_retries(task_id) + 1)

        candidates.sort(key=compute_weight, reverse=True)
