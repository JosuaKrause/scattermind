from scattermind.system.base import TaskId
from scattermind.system.client.client import ClientPool


class NodeStrategy:
    # FIXME break down pressure into components
    def own_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        raise NotImplementedError()

    def other_score(
            self,
            *,
            queue_length: int,
            pressure: float,
            expected_pressure: float,
            cost_to_load: float) -> float:
        raise NotImplementedError()

    def want_to_switch(self, own_score: float, other_score: float) -> bool:
        raise NotImplementedError()


class QueueStrategy:  # pylint: disable=too-few-public-methods
    # FIXME implement strategies using symbolic redis
    def sort_queue(
            self,
            cpool: ClientPool,
            candidates: list[TaskId]) -> None:
        raise NotImplementedError()
