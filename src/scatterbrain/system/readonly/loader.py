from typing import Literal, TypedDict

from scatterbrain.system.plugins import load_plugin
from scatterbrain.system.readonly.access import ReadonlyAccess


LocalReadonlyAccessModule = TypedDict('LocalReadonlyAccessModule', {
    "name": Literal["ram"],
})


ReadonlyAccessModule = LocalReadonlyAccessModule


def load_readonly_access(module: ReadonlyAccessModule) -> ReadonlyAccess:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ReadonlyAccess, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "ram":
        from scatterbrain.system.readonly.ram import RAMAccess
        return RAMAccess()
    raise ValueError(f"unknown readonly access: {module['name']}")
