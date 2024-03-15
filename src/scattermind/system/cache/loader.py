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
"""Loads a graph cache."""
from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.cache.cache import GraphCache
from scattermind.system.plugins import load_plugin


NoCacheModule = TypedDict('NoCacheModule', {
    "name": Literal["nocache"],
})
"""No graph caching."""
RedisCacheModule = TypedDict('RedisCacheModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})
"""Cache using redis."""


GraphCacheModule = NoCacheModule | RedisCacheModule
"""A graph cache module configuration."""


def load_graph_cache(module: GraphCacheModule) -> GraphCache:
    """
    Load a graph cache from a module configuration. The `name` key can be a
    valid python module path to load a plugin.

    Args:
        module (GraphCacheModule): The module configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        GraphCache: The graph cache.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(GraphCache, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "nocache":
        from scattermind.system.cache.nocache import NoCache
        return NoCache()
    if module["name"] == "redis":
        from scattermind.system.cache.redis import RedisCache
        return RedisCache(module["cfg"])
    raise ValueError(f"unknown graph cache: {module['name']}")
