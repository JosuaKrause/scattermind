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
