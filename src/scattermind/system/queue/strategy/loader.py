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
"""Loads strategies."""
from typing import Literal, TypedDict

from scattermind.system.plugins import load_plugin
from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    QueueStrategy,
)


SimpleNodeStrategyModule = TypedDict('SimpleNodeStrategyModule', {
    "name": Literal["simple"],
})
"""Configuration for a simple node strategy."""


DedicatedNodeStrategyModule = TypedDict('DedicatedNodeStrategyModule', {
    "name": Literal["dedicated"],
})
"""Configuration for a dedicated node strategy."""


NodeStrategyModule = SimpleNodeStrategyModule | DedicatedNodeStrategyModule
"""Configuration for node strategies."""


def load_node_strategy(module: NodeStrategyModule) -> NodeStrategy:
    """
    Loads a node strategy.

    Args:
        module (NodeStrategyModule): The configuration. If `name` is a fully
            qualified python module it will be loaded as plugin.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        NodeStrategy: The node strategy.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(NodeStrategy, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "simple":
        from scattermind.system.queue.strategy.node.simple import (
            SimpleNodeStrategy,
        )
        return SimpleNodeStrategy()
    if module["name"] == "dedicated":
        from scattermind.system.queue.strategy.node.dedicated import (
            DedicatedNodeStrategy,
        )
        return DedicatedNodeStrategy()
    raise ValueError(f"unknown node strategy: {module['name']}")


SimpleQueueStrategyModule = TypedDict('SimpleQueueStrategyModule', {
    "name": Literal["simple"],
})
"""Configuration for a simple queue strategy."""


QueueStrategyModule = SimpleQueueStrategyModule
"""Configuration for queue strategies."""


def load_queue_strategy(module: QueueStrategyModule) -> QueueStrategy:
    """
    Loads a queue strategy.

    Args:
        module (QueueStrategyModule): The configuration. If `name` is a fully
            qualified python module it will be loaded as plugin.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        QueueStrategy: The queue strategy.
    """
    # pylint: disable=import-outside-toplevel
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
