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
"""Loads a client pool."""
from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.client.client import ClientPool
from scattermind.system.plugins import load_plugin


LocalClientPoolModule = TypedDict('LocalClientPoolModule', {
    "name": Literal["local"],
})
"""RAM-only client pool module configuration."""
RedisClientPoolModule = TypedDict('RedisClientPoolModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})
"""Redis base client pool module configuration. `cfg` are the redis connection
settings."""


ClientPoolModule = LocalClientPoolModule | RedisClientPoolModule
"""A client pool module configuration."""


def load_client_pool(module: ClientPoolModule) -> ClientPool:
    """
    Load a client pool from a module configuration. The `name` key can be a
    valid python module path to load a plugin.

    Args:
        module (ClientPoolModule): The module configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        ClientPool: The client pool.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ClientPool, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scattermind.system.client.local import LocalClientPool
        return LocalClientPool()
    if module["name"] == "redis":
        from scattermind.system.client.redis import RedisClientPool
        return RedisClientPool(module["cfg"])
    raise ValueError(f"unknown client pool: {module['name']}")
