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

from typing_extensions import NotRequired

from scattermind.system.logger.log import EventListener
from scattermind.system.plugins import load_plugin


StdoutListenerDef = TypedDict('StdoutListenerDef', {
    "name": Literal["stdout"],
    "show_debug": NotRequired[bool],
})


EventListenerDef = StdoutListenerDef


def load_event_listener(
        eldef: EventListenerDef, disable_events: list[str]) -> EventListener:
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
