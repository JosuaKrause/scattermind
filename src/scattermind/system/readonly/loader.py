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
"""Create a readonly access."""
from typing import Literal, TypedDict

from scattermind.system.plugins import load_plugin
from scattermind.system.readonly.access import ReadonlyAccess


LocalReadonlyAccessModule = TypedDict('LocalReadonlyAccessModule', {
    "name": Literal["ram"],
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
        return RAMAccess()
    raise ValueError(f"unknown readonly access: {module['name']}")
