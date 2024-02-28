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
"""Provides functionality to allow loading custom implementations via plugin.
"""
import importlib
from typing import TypeVar

from scattermind.system.util import full_name


T = TypeVar('T')


PLUGIN_CACHE: dict[str, dict[str, type]] = {}
"""Caches previously loaded types."""


def load_plugin(base: type[T], name: str) -> type[T]:
    """
    Loads custom code as plugin. For a plugin to be loadable it must be visible
    as python module to the current process. The module must contain exactly
    one sub-class of the base class. Other classes and symbols are allowed.

    Args:
        base (type[T]): The expected base type.
        name (str): The fully qualified name of the plugin to load. The plugin
            must be accessible as python module from the cwd.

    Raises:
        ValueError: If the module cannot be loaded. There might be no sub-class
            of the base class in the module or there might be multiple
            sub-classes.

    Returns:
        type[T]: The loaded plugin.
    """
    base_name = full_name(base)
    base_cache = PLUGIN_CACHE.get(base_name)
    if base_cache is None:
        base_cache = {}
        PLUGIN_CACHE[base_name] = base_cache
    res = base_cache.get(name)
    if res is not None:
        return res
    mod = importlib.import_module(name)
    candidates = [
        cls
        for cls in mod.__dict__.values()
        if isinstance(cls, type)
        and cls.__module__ == name
        and issubclass(cls, base)
    ]
    if len(candidates) != 1:
        cands = [
            can.__name__ for can in candidates
        ]
        raise ValueError(
            f"ambiguous or missing plugin for {base_name}: {cands}")
    res = candidates[0]
    base_cache[name] = res
    return res
