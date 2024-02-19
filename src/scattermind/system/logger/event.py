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
"""Event types for the logging system."""
import datetime
from typing import Literal, NotRequired, TypedDict

from scattermind.system.base import TaskId
from scattermind.system.logger.context import ContextInfo
from scattermind.system.logger.error import ErrorCode
from scattermind.system.names import NName


ErrorEvent = TypedDict('ErrorEvent', {
    "name": Literal["error"],
    "message": str,
    "traceback": list[str],
    "code": ErrorCode,
})
"""An event to report errors."""


RetryEvent = TypedDict('RetryEvent', {
    "name": Literal["retry"],
    "message": str,
    "traceback": list[str],
    "code": ErrorCode,
    "retries": int,
})
"""Event to report that a task had to retry processing. This event can contain
an error that triggered the retry."""


WarningEvent = TypedDict('WarningEvent', {
    "name": Literal["warning"],
    "message": str,
})
"""A event to report warnings."""


TaskEvent = TypedDict('TaskEvent', {
    "name": Literal["tasks"],
    "tasks": list[TaskId],
})
"""An event affecting a collection of tasks."""


NodeEvent = TypedDict('NodeEvent', {
    "name": Literal["node"],
    "action": Literal["load", "load_done", "unload"],
    "target": NName,
})
"""Event to indicate that a node has been loaded or unloaded."""


OutputEvent = TypedDict('OutputEvent', {
    "name": Literal["output"],
    "entry": str,
    "stdout": str,
    "stderr": str,
})
"""Event collecting writes to stdout or stderr."""


QueueMeasureEvent = TypedDict('QueueMeasureEvent', {
    "name": Literal["queue_input"],
    "length": NotRequired[int],
    "weight": NotRequired[float],
    "pressure": NotRequired[float],
    "expected_pressure": NotRequired[float],
    "cost": NotRequired[float],
    "claimants": NotRequired[int],
    "loaded": NotRequired[int],
    "picked": NotRequired[bool],
})
"""Event to report comparing two queues."""


ExecutorEvent = TypedDict('ExecutorEvent', {
    "name": Literal["executor"],
    "action": Literal["start", "stop", "reclaim"],
    "executors": NotRequired[int],
    "listeners": NotRequired[int],
})
"""Event to indicate that an executor has been started or stopped. A stop event
might not always be fired."""


AnyEvent = (
    ErrorEvent
    | RetryEvent
    | WarningEvent
    | TaskEvent
    | NodeEvent
    | OutputEvent
    | QueueMeasureEvent
    | ExecutorEvent
)
"""An event that can be logged to the event stream."""


EventInfo = TypedDict('EventInfo', {
    "when": datetime.datetime,
    "name": str,
    "ctx": ContextInfo,
    "event": AnyEvent,
})
"""Full information and context for events that can be logged to the event
stream."""
