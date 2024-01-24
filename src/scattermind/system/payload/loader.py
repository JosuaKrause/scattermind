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

from redipy import RedisConfig

from scattermind.system.payload.data import DataStore
from scattermind.system.plugins import load_plugin
from scattermind.system.redis_util import DataMode


LocalDataStoreModule = TypedDict('LocalDataStoreModule', {
    "name": Literal["local"],
    "max_size": int,
})
RedisDataStoreModule = TypedDict('RedisDataStoreModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
    "mode": DataMode,
})


DataStoreModule = LocalDataStoreModule | RedisDataStoreModule


def load_store(module: DataStoreModule) -> DataStore:
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
