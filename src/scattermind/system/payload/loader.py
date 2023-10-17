from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.payload.data import DataStore
from scattermind.system.plugins import load_plugin


LocalDataStoreModule = TypedDict('LocalDataStoreModule', {
    "name": Literal["local"],
    "max_size": int,
})
RedisDataStoreModule = TypedDict('RedisDataStoreModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})


DataStoreModule = LocalDataStoreModule


def load_store(module: LocalDataStoreModule) -> DataStore:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(DataStore, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scattermind.system.payload.local import LocalDataStore
        return LocalDataStore(module["max_size"])
    if module["name"] == "redis":
        from scattermind.system.payload.redis import RedisDataStore
        return RedisDataStore(module["cfg"])
    raise ValueError(f"unknown data store: {module['name']}")
