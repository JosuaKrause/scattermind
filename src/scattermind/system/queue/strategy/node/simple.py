from scattermind.system.queue.strategy.strategy import NodeStrategy


class SimpleNodeStrategy(NodeStrategy):
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
