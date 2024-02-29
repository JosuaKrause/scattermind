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
