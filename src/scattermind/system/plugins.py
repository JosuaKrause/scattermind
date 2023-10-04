import importlib
from typing import TypeVar

from scattermind.system.util import full_name


T = TypeVar('T')


PLUGIN_CACHE: dict[str, dict[str, type]] = {}


def load_plugin(base: type[T], name: str) -> type[T]:
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
