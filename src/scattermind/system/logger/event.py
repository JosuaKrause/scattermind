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
import datetime
from typing import Literal, TypedDict

from scattermind.system.base import TaskId
from scattermind.system.logger.context import ContextInfo
from scattermind.system.logger.error import ErrorCode


ErrorEvent = TypedDict('ErrorEvent', {
    "name": Literal["error"],
    "message": str,
    "traceback": list[str],
    "code": ErrorCode,
})


RetryEvent = TypedDict('RetryEvent', {
    "name": Literal["retry"],
    "message": str,
    "traceback": list[str],
    "code": ErrorCode,
    "retries": int,
})


WarningEvent = TypedDict('WarningEvent', {
    "name": Literal["warning"],
    "message": str,
})


TaskEvent = TypedDict('TaskEvent', {
    "name": Literal["tasks"],
    "tasks": list[TaskId],
})


NodeEvent = TypedDict('NodeEvent', {
    "name": Literal["node"],
    "action": Literal["load", "unload"],
})


OutputEvent = TypedDict('OutputEvent', {
    "name": Literal["output"],
    "entry": str,
    "stdout": str,
    "stderr": str,
})


QueueMeasureEvent = TypedDict('QueueMeasureEvent', {
    "name": Literal["queue_input"],
    "length": int,
    "pressure": float,
    "expected_pressure": float,
    "score": float,
})


ExecutorEvent = TypedDict('ExecutorEvent', {
    "name": Literal["executor"],
    "action": Literal["start", "stop"],
})


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


EventInfo = TypedDict('EventInfo', {
    "when": datetime.datetime,
    "name": str,
    "ctx": ContextInfo,
    "event": AnyEvent,
})
