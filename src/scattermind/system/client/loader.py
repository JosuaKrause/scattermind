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
