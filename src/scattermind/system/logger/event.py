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
