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
"""Event types for the logging system."""
import datetime
from typing import Literal, NotRequired, TypedDict

from scattermind.system.base import TaskId
from scattermind.system.logger.context import ContextInfo
from scattermind.system.logger.error import ErrorCode
from scattermind.system.names import QualifiedNodeName


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


GhostTaskEvent = TypedDict('GhostTaskEvent', {
    "name": Literal["ghost"],
    "message": str,
    "task": TaskId,
})


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
    "target": QualifiedNodeName,
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
    "action": Literal[
        "start",
        "stop",
        "reclaim",
        "reclaim_start",
        "reclaim_stop",
        "heartbeat_start",
        "heartbeat_stop"],
    "executors": NotRequired[int],
    "listeners": NotRequired[int],
})
"""Event to indicate that an executor has been started or stopped. A stop event
might not always be fired."""


AnyEvent = (
    ErrorEvent
    | RetryEvent
    | GhostTaskEvent
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
