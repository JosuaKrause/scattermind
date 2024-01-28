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
"""Loads a queue pool."""
from typing import Literal, TypedDict

from redipy import RedisConfig
from typing_extensions import NotRequired

from scattermind.system.plugins import load_plugin
from scattermind.system.queue.queue import QueuePool


LocalQueuePoolModule = TypedDict('LocalQueuePoolModule', {
    "name": Literal["local"],
    "check_assertions": NotRequired[bool],
})
"""Configuration for RAM-only queue pools. `check_assertions` is for activating
assertions (which queue is a task in?)."""
RedisQueuePoolModule = TypedDict('RedisQueuePoolModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
    "check_assertions": NotRequired[bool],
})
"""Configuration for a redis queue pool. `cfg` are the redis connection
settings. `check_assertions` is for activating assertions (which queue is a
task in?)."""


QueuePoolModule = LocalQueuePoolModule | RedisQueuePoolModule
"""Queue pool configuration."""


def load_queue_pool(module: QueuePoolModule) -> QueuePool:
    """
    Loads a queue pool from a given configuration. `name` can be python module
    fully qualified name instead to load a queue pool via plugin.

    Args:
        module (QueuePoolModule): The configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        QueuePool: The queue pool.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(QueuePool, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scattermind.system.queue.local import LocalQueuePool
        return LocalQueuePool(
            check_assertions=module.get("check_assertions", False))
    if module["name"] == "redis":
        from scattermind.system.queue.redis import RedisQueuePool
        return RedisQueuePool(
            cfg=module["cfg"],
            check_assertions=module.get("check_assertions", False))
    raise ValueError(f"unknown queue pool: {module['name']}")
