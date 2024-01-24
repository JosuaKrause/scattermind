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
