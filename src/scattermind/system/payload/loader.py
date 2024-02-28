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
