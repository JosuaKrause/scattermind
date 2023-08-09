from typing import Literal, TypedDict

from scatterbrain.system.client.client import ClientPool
from scatterbrain.system.plugins import load_plugin


LocalClientPoolModule = TypedDict('LocalClientPoolModule', {
    "name": Literal["local"],
})


ClientPoolModule = LocalClientPoolModule


def load_client_pool(module: ClientPoolModule) -> ClientPool:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ClientPool, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scatterbrain.system.client.local import LocalClientPool
        return LocalClientPool()
    raise ValueError(f"unknown client pool: {module['name']}")
