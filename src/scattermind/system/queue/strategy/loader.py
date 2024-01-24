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
