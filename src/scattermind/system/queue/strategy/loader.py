from typing import Literal, TypedDict

from scattermind.system.plugins import load_plugin
from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    QueueStrategy,
)


SimpleNodeStrategyModule = TypedDict('SimpleNodeStrategyModule', {
    "name": Literal["simple"],
})


NodeStrategyModule = SimpleNodeStrategyModule


def load_node_strategy(module: NodeStrategyModule) -> NodeStrategy:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(NodeStrategy, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "simple":
        from scattermind.system.queue.strategy.node.simple import (
            SimpleNodeStrategy,
        )
        return SimpleNodeStrategy()
    raise ValueError(f"unknown node strategy: {module['name']}")


SimpleQueueStrategyModule = TypedDict('SimpleQueueStrategyModule', {
    "name": Literal["simple"],
})


QueueStrategyModule = SimpleQueueStrategyModule


def load_queue_strategy(module: QueueStrategyModule) -> QueueStrategy:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(QueueStrategy, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "simple":
        from scattermind.system.queue.strategy.queue.simple import (
            SimpleQueueStrategy,
        )
        return SimpleQueueStrategy()
    raise ValueError(f"unknown queue strategy: {module['name']}")
