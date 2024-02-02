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
"""Provides the custom logging for scattermind. Logging can be both messages,
as well as, statistics about the runtime. Different backends can be used to
compile statistics and provide monitoring capabilities."""
import contextlib

# import io
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
    """The base class for event listeners."""
    def __init__(self, *, disable_events: list[str] | None = None) -> None:
        """
        Creates an event listener. Sub-classes need to expose all keyword
        arguments as well.

        Args:
            disable_events (list[str] | None, optional): Which events to
                ignore. Each string is a plain text pattern and must fully
                match event name segments. By default only prefixes are
                checked. A `.` prefix enables the rule to be applied to inner
                matches as well. A `!` prefix negates the rule and makes it
                inclusive instead. A value of None allows all events. Defaults
                to None.

        Raises:
            ValueError: If an invalid filter was passed.
        """
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
        """
        Whether the event listeners wants to capture a given event. By default
        this method returns True for all events, capturing all events.
        Sub-classes must override this method if a different behavior is
        preferred.

        Args:
            name (str): The event name. Event names are hierarchical segments
                joined by `.`.

        Returns:
            bool: True, if the event should be processed by this event
                listener.
        """
        return True  # NOTE: overwrite in subclass

    def capture_event(self, name: str) -> bool:
        """
        Whether the event listeners captures a given event. If you want to
        change the behavior override py::method:`do_capture_static`.

        Args:
            name (str): THe event name. Event names are hierarchical segments
                joined by `.`.

        Returns:
            bool: True, if the event will be processed by this event listener.
        """
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
        """
        Processes a log event.

        Args:
            event (EventInfo): The event to process.
        """
        raise NotImplementedError()


class EventStream:
    """An event stream for logging and capturing runtime statistics. `log`
    methods can be used to add events to the stream. The
    ::py:method:`add_listener` method is used to add event listeners, i.e.,
    processing backends."""
    def __init__(self) -> None:
        """
        Creates a new event stream.
        """
        self._listeners: list[EventListener] = []
        self._reported_warnings: set[str] = set()

    def add_listener(self, listener: EventListener) -> None:
        """
        Add an event listener, i.e., processing backend.

        Args:
            listener (EventListener): The listener.
        """
        self._listeners.append(listener)

    @contextlib.contextmanager
    def log_output(self, name: str, entry: str) -> Iterator[None]:
        """
        A contextmanager to redirect all stdout and stderr content to events
        for the given resource block.

        Args:
            name (str): The name of the event. Hierarchical segments joined by
                `.`. Names for this function should start with `output.`.
            entry (str): Message to provide further context.
        """
        # pylint: disable=unused-argument
        # stdout = io.StringIO()
        # stderr = io.StringIO()
        # try:
        #     with (
        #             contextlib.redirect_stdout(stdout),
        #             contextlib.redirect_stderr(stderr)):
        #         yield
        # finally:
        #     if stdout.tell() > 0 or stderr.tell() > 0:
        #         self.log_event(
        #             name,
        #             {
        #                 "name": "output",
        #                 "entry": entry,
        #                 "stdout": stdout.getvalue(),
        #                 "stderr": stderr.getvalue(),
        #             })
        yield  # FIXME: make it work with multiple threads

    def log_event(
            self,
            name: str,
            event: AnyEvent,
            *,
            is_error_ctx: bool = False,
            adjust_ctx: ContextInfo | None = None) -> None:
        """
        Log an event. This is a convenience function for when the event is
        quick to create (does not require costly computation to gather
        information) or already exists anyway.

        Args:
            name (str): The name of the event. Hierarchical segments joined by
                `.`.
            event (AnyEvent): The event.
            is_error_ctx (bool, optional): Whether the current error context
                should be included with the event. Defaults to False.
            adjust_ctx (ContextInfo | None, optional): Additional context
                information to be included in the event. Defaults to None.
        """
        self.log_lazy(
            name,
            lambda: event,
            is_error_ctx=is_error_ctx,
            adjust_ctx=adjust_ctx)

    def log_warning(self, name: str, message: str) -> None:
        """
        Log a warning. The same warning (by name) is only reported once per
        event stream instance.

        Args:
            name (str): The name of the event. Hierarchical segments joined by
                `.`.
            message (str): The warning message to provide more details.
        """
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
        """
        Log an error.

        Args:
            name (str): The name of the event. Hierarchical segments joined by
                `.`.
            code (ErrorCode): The error code.
            exc (BaseException | None, optional): The exception that caused
                the error. If set to None the exception will be inferred from
                the context. Defaults to None.
        """
        tback = [
            line.rstrip()
            for line in (
                traceback.format_exc().splitlines()
                if exc is None
                else traceback.format_exception(exc))
        ]
        if exc is None:
            exc = sys.exception()
        notes_val = None if exc is None else getattr(exc, "__notes__", None)
        if notes_val:
            notes = f"\n{notes_val}"
        else:
            notes = ""
        message = f"{exc}{notes}"
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
        """
        Lazily log an event. This is the preferred method to log an event.
        The event will only be created if an event listener backend actually
        wants to process the event.

        Args:
            name (str): The name of the event. Hierarchical segments joined by
                `.`.
            event_gen (Callable[[], AnyEvent]): A callback that creates the
                event when needed. Information that is only required for the
                event should be retrieved in this callback to avoid computation
                when the event is not processed.
            is_error_ctx (bool, optional): Whether the current error context
                should be included with the event. Defaults to False.
            adjust_ctx (ContextInfo | None, optional): Additional context
                information to be included in the event. Defaults to None.
        """
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
