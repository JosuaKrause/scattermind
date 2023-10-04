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
