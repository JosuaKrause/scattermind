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
