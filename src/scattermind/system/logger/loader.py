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
