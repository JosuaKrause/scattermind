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
"""Prints all events to stdout."""
from typing import Any

from scattermind.system.logger.context import ctx_format
from scattermind.system.logger.event import EventInfo
from scattermind.system.logger.log import EventListener
from scattermind.system.util import fmt_time


class StdoutListener(EventListener):
    """Prints all events to stdout."""
    def __init__(
            self,
            *,
            disable_events: list[str] | None = None,
            show_debug: bool = False) -> None:
        super().__init__(disable_events=disable_events)
        self._show_debug = show_debug

    def do_capture_static(
            self,
            name: str) -> bool:  # pylint: disable=unused-argument
        if self._show_debug:
            return True
        return not name.startswith("debug.")

    def log_event(self, event: EventInfo) -> None:
        time_str = fmt_time(event["when"])
        name = event["name"]
        ctx = ctx_format(event["ctx"])

        def render_value(key: str, value: Any) -> str:
            if key == "traceback":
                return "\n".join(value)
            return f"{value}"

        evt = ", ".join(
            f"{key}={render_value(key, value)}"
            for key, value in event["event"].items())
        print(f"{time_str} {ctx} {name}: {evt}".rstrip())
