from typing import Literal, TypedDict

from scattermind.system.payload.data import DataStore
from scattermind.system.plugins import load_plugin


LocalDataStoreModule = TypedDict('LocalDataStoreModule', {
    "name": Literal["local"],
    "max_size": int,
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
    raise ValueError(f"unknown data store: {module['name']}")
