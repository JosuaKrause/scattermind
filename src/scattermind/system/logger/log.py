import contextlib
import io
import sys
import traceback
from collections.abc import Callable, Iterator

from scattermind.system.logger.context import (
    ContextInfo,
    get_ctx,
    get_preexc_ctx,
)
from scattermind.system.logger.error import ErrorCode
from scattermind.system.logger.event import AnyEvent, EventInfo
from scattermind.system.util import is_partial_match, now


class EventListener:
    def __init__(self, *, disable_events: list[str] | None = None) -> None:
        self._pos_filters = []
        self._neg_filters = []
        all_filters = [] if disable_events is None else disable_events
        for cur in all_filters:
            if not cur or cur in ["", ".", "!", "!."]:
                raise ValueError(f"invalid filter: {cur}")
            flipped = cur.removeprefix("!")
            if cur == flipped:
                self._neg_filters.append(cur)
            else:
                self._pos_filters.append(flipped)

    def do_capture_static(
            self,
            name: str) -> bool:  # pylint: disable=unused-argument
        return True  # NOTE: overwrite in subclass

    def capture_event(self, name: str) -> bool:
        if not self.do_capture_static(name):
            return False
        for filter_str in self._pos_filters:
            if is_partial_match(name, filter_str):
                return True
        for filter_str in self._neg_filters:
            if is_partial_match(name, filter_str):
                return False
        return True

    def log_event(self, event: EventInfo) -> None:
        raise NotImplementedError()


class EventStream:
    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._reported_warnings: set[str] = set()

    def add_listener(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    @contextlib.contextmanager
    def log_output(self, name: str, entry: str) -> Iterator[None]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr)):
            yield
        if stdout.tell() > 0 or stderr.tell() > 0:
            self.log_event(
                name,
                {
                    "name": "output",
                    "entry": entry,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                })

    def log_event(
            self,
            name: str,
            event: AnyEvent,
            *,
            is_error_ctx: bool = False,
            adjust_ctx: ContextInfo | None = None) -> None:
        self.log_lazy(
            name,
            lambda: event,
            is_error_ctx=is_error_ctx,
            adjust_ctx=adjust_ctx)

    def log_warning(self, name: str, message: str) -> None:
        self.log_event(
            name,
            {
                "name": "warning",
                "message": message,
            })

    def log_error(
            self,
            name: str,
            code: ErrorCode,
            *,
            exc: BaseException | None = None) -> None:
        tback = [
            line.rstrip()
            for line in (
                traceback.format_exc()
                if exc is None
                else traceback.format_exception(exc))
        ]
        message = f"{sys.exception()}"
        self.log_event(
            name,
            {
                "name": "error",
                "code": code,
                "message": message,
                "traceback": tback,
            },
            is_error_ctx=True)

    def log_lazy(
            self,
            name: str,
            event_gen: Callable[[], AnyEvent],
            *,
            is_error_ctx: bool = False,
            adjust_ctx: ContextInfo | None = None) -> None:
        event_info: EventInfo | None = None
        for listener in self._listeners:
            if not listener.capture_event(name):
                continue
            if event_info is None:
                when = now()
                ctx = get_preexc_ctx() if is_error_ctx else get_ctx()
                if adjust_ctx is not None:
                    ctx.update(adjust_ctx)
                event = event_gen()
                if event["name"] == "warning":
                    if name in self._reported_warnings:
                        return  # report a warning only once per instance
                    self._reported_warnings.add(name)
                event_info = {
                    "when": when,
                    "ctx": ctx,
                    "name": name,
                    "event": event,
                }
            listener.log_event(event_info)
