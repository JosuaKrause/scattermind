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
"""Loads event stream listeners, i.e., logging backends."""
from typing import Literal, TypedDict

from typing_extensions import NotRequired

from scattermind.system.logger.log import EventListener
from scattermind.system.plugins import load_plugin


StdoutListenerDef = TypedDict('StdoutListenerDef', {
    "name": Literal["stdout"],
    "show_debug": NotRequired[bool],
})
"""Listens to (almost) all events and prints them to stdout."""


EventListenerDef = StdoutListenerDef
"""Event listener configurations."""


def load_event_listener(
        eldef: EventListenerDef, disable_events: list[str]) -> EventListener:
    """
    Load the event listener for the given configuration. If `name` is set to
    a fully qualified python module the listener is loaded as plugin.

    Args:
        eldef (EventListenerDef): The configuration.
        disable_events (list[str]): Which events to ignore. Each string is a
            plain text pattern and must fully match event name segments.
            By default only prefixes are checked. A `.` prefix enables the rule
            to be applied to inner matches as well. A `!` prefix negates the
            rule and makes it inclusive instead.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        EventListener: The event listener.
    """
    # pylint: disable=import-outside-toplevel
    if "." in eldef["name"]:
        kwargs = dict(eldef)
        plugin = load_plugin(EventListener, f"{kwargs.pop('name')}")
        return plugin(disable_events=disable_events, **kwargs)
    if eldef["name"] == "stdout":
        from scattermind.system.logger.listeners.stdout import StdoutListener
        return StdoutListener(
            disable_events=disable_events,
            show_debug=eldef.get("show_debug", False))
    raise ValueError(f"unknown event listener: {eldef['name']}")
