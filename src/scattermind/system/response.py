from typing import Literal, TYPE_CHECKING, TypedDict

from scattermind.system.logger.error import ErrorInfo, raise_error, warn_error


if TYPE_CHECKING:
    from scattermind.system.payload.values import TaskValueContainer


TaskStatus = Literal[
    "init",
    "wait",
    "busy",
    "ready",
    "done",
    "error",
    "unknown",
]
TASK_STATUS_INIT: TaskStatus = "init"
TASK_STATUS_WAIT: TaskStatus = "wait"
TASK_STATUS_BUSY: TaskStatus = "busy"
TASK_STATUS_READY: TaskStatus = "ready"
TASK_STATUS_DONE: TaskStatus = "done"
TASK_STATUS_ERROR: TaskStatus = "error"
TASK_STATUS_UNKNOWN: TaskStatus = "unknown"


ResponseObject = TypedDict('ResponseObject', {
    "status": TaskStatus,
    "result": 'TaskValueContainer | None',
    "duration": float,
    "retries": int,
    "error": ErrorInfo | None,
})


def response_ok(response: ResponseObject, *, no_warn: bool = False) -> None:
    if response["error"] is not None:
        if response["status"] != "error" and not no_warn:
            warn_error(response["error"], response["retries"])
        else:
            raise_error(response["error"], response["retries"])
