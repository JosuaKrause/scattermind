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
"""Loads a payload data store from a given configuration."""
from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.payload.data import DataStore
from scattermind.system.plugins import load_plugin
from scattermind.system.redis_util import DataMode


LocalDataStoreModule = TypedDict('LocalDataStoreModule', {
    "name": Literal["local"],
    "max_size": int,
})
"""An in-memory payload data store. The number of entries is limited via
`max_size`. The oldest entries get purged first."""
RedisDataStoreModule = TypedDict('RedisDataStoreModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
    "mode": DataMode,
})
"""A redis based payload data store. `mode` decides the cache freeing strategy
and `cfg` defines the redis connection settings."""


DataStoreModule = LocalDataStoreModule | RedisDataStoreModule
"""Configuration of a payload data store."""


def load_store(module: DataStoreModule) -> DataStore:
    """
    Loads the payload data store for the given configuration.

    Args:
        module (DataStoreModule): The data store configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        DataStore: The data store.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(DataStore, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scattermind.system.payload.local import LocalDataStore
        return LocalDataStore(module["max_size"])
    if module["name"] == "redis":
        from scattermind.system.payload.redis import RedisDataStore
        return RedisDataStore(module["cfg"], module["mode"])
    raise ValueError(f"unknown data store: {module['name']}")
