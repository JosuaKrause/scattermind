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
