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
"""Create a readonly access."""
from typing import Literal, TypedDict

from scattermind.system.plugins import load_plugin
from scattermind.system.readonly.access import ReadonlyAccess


LocalReadonlyAccessModule = TypedDict('LocalReadonlyAccessModule', {
    "name": Literal["ram"],
    "scratch": str,
})
"""A RAM based readonly access. This can only be used locally and is used
mostly for tests."""


ReadonlyAccessModule = LocalReadonlyAccessModule
"""Info for creating a readonly access."""


def load_readonly_access(module: ReadonlyAccessModule) -> ReadonlyAccess:
    """
    Creates a readonly access.

    Args:
        module (ReadonlyAccessModule): The info to load the access. The name
            can be a predefined module or a plugin module by specifying a
            fully qualified python module path.

    Raises:
        ValueError: If the module could not be loaded.

    Returns:
        ReadonlyAccess: The readonly access.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ReadonlyAccess, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "ram":
        from scattermind.system.readonly.ram import RAMAccess
        return RAMAccess(module["scratch"])
    raise ValueError(f"unknown readonly access: {module['name']}")
