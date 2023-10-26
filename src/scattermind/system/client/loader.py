from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.client.client import ClientPool
from scattermind.system.plugins import load_plugin


LocalClientPoolModule = TypedDict('LocalClientPoolModule', {
    "name": Literal["local"],
})
RedisClientPoolModule = TypedDict('RedisClientPoolModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})


ClientPoolModule = LocalClientPoolModule | RedisClientPoolModule


def load_client_pool(module: ClientPoolModule) -> ClientPool:
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
