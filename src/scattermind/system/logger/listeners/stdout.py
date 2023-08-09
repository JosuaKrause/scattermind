from scattermind.system.logger.context import ctx_format
from scattermind.system.logger.event import EventInfo
from scattermind.system.logger.log import EventListener
from scattermind.system.util import fmt_time


class StdoutListener(EventListener):
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
        evt = ", ".join(
            f"{key}={value}"
            for key, value in event["event"].items())
        print(f"{time_str} {ctx} {name}: {evt}".rstrip())
