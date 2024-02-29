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
