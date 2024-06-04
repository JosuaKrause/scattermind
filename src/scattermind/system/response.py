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
"""Defines how a task response contains."""
from typing import cast, get_args, Literal, TypedDict

from scattermind.system.info import DataFormat, UserDataFormatJSON
from scattermind.system.logger.error import (
    ErrorInfo,
    ErrorJSON,
    from_error_json,
    raise_error,
    to_error_json,
    warn_error,
)
from scattermind.system.names import GNamespace
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.redis_util import (
    redis_to_robj,
    redis_to_tvc,
    robj_to_redis,
    tvc_to_redis,
)


TaskStatus = Literal[
    "init",
    "wait",
    "busy",
    "ready",
    "done",
    "deferred",
    "error",
    "unknown",
]
"""The status of the task."""
TASK_STATUS_ALL: set[TaskStatus] = set(get_args(TaskStatus))
"""All valid task status values."""
TASK_STATUS_INIT: TaskStatus = "init"
"""The task has just been created."""
TASK_STATUS_WAIT: TaskStatus = "wait"
"""The task is waiting to continue computation."""
TASK_STATUS_BUSY: TaskStatus = "busy"
"""The task is busy computing."""
TASK_STATUS_READY: TaskStatus = "ready"
"""The results of the task are ready to be read."""
TASK_STATUS_DONE: TaskStatus = "done"
"""The results have been read and the task can be cleaned up."""
TASK_STATUS_DEFER: TaskStatus = "deferred"
"""The task is computed by a different task via caching."""
TASK_STATUS_ERROR: TaskStatus = "error"
"""An error occured while executing the task."""
TASK_STATUS_UNKNOWN: TaskStatus = "unknown"
"""The status of the task is unknown. The task might not exist."""


TASK_COMPLETE: set[TaskStatus] = {
    TASK_STATUS_READY,
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_UNKNOWN,
}
"""The task has either finished, had an error, or is not known."""


def to_status(text: str) -> TaskStatus:
    """
    Converts a string into a task status.

    Args:
        text (str): The string.

    Raises:
        ValueError: If the string does not represent a valid task status.

    Returns:
        TaskStatus: The task status.
    """
    if text not in TASK_STATUS_ALL:
        raise ValueError(f"invalid status {text}")
    return cast(TaskStatus, text)


ResponseObject = TypedDict('ResponseObject', {
    "ns": GNamespace | None,
    "status": TaskStatus,
    "result": TaskValueContainer | None,
    "duration": float,
    "retries": int,
    "error": ErrorInfo | None,
    "fmt": DataFormat | None,
})
"""
Information about a task. The status indicates the progress of the task's
computation. The result contains the final result of the task once it has
completed the computation. The duration indicates the time it took for the task
to complete in seconds. The value might not be correct if the task has not
finished yet. The number of retries indicate how many times the task
computation needed to be restarted. The error contains a value if the task
failed to compute.
"""


def response_ok(response: ResponseObject, *, no_warn: bool = False) -> None:
    """
    Checks whether a task is still alive or has completed successfully. If the
    task failed a ValueError will be raised. If the task was eventually
    successful but had runs that resulted in errors a warning is emitted
    indicating the last encountered error. If no_warn is set a ValueError will
    be raised in this case as well.

    Args:
        response (ResponseObject): The information about the task.
        no_warn (bool, optional): If set a ValueError is raised when otherwise
            a warning would be generated. Defaults to False.

    Raises:
        ValueError: If the task failed or if no_warn is True and an error
            happened during the computation of the task but another run was
            successful.
    """
    if response["error"] is not None:
        if response["status"] != "error" and not no_warn:
            warn_error(response["error"], response["retries"])
        else:
            raise_error(response["error"], response["retries"])


def response_to_redis(response: ResponseObject) -> str:
    """
    Convert a response into a redis storable format.

    Args:
        response (ResponseObject): The response object.

    Returns:
        str: Redis storable format.
    """
    result = response["result"]
    error = response["error"]
    fmt = response["fmt"]
    obj: dict[str, UserDataFormatJSON | ErrorJSON | str | None] = {
        "ns": None if response["ns"] is None else response["ns"].get(),
        "status": response["status"],
        "result": None if result is None else tvc_to_redis(result),
        "duration": f"{response['duration']}",
        "retries": f"{response['retries']}",
        "error": None if error is None else to_error_json(error),
        "fmt": None if fmt is None else fmt.data_format_to_json(),
    }
    return robj_to_redis(obj)


def redis_to_response(text: str) -> ResponseObject:
    """
    Convert a value from redis back into a response object.

    Args:
        text (str): The redis value.

    Returns:
        ResponseObject: The response object.
    """
    obj = redis_to_robj(text)
    result = obj["result"]
    error = obj["error"]
    fmt = obj["fmt"]
    if fmt is None or result is None:
        data_format = None
        result_tvc = None
    else:
        data_format = DataFormat.data_format_from_json(fmt)
        result_tvc = redis_to_tvc(result, data_format)
    return {
        "ns": None if obj["ns"] is None else GNamespace(obj["ns"]),
        "status": to_status(obj["status"]),
        "result": result_tvc,
        "duration": float(obj["duration"]),
        "retries": int(obj["retries"]),
        "error": None if error is None else from_error_json(error),
        "fmt": data_format,
    }
