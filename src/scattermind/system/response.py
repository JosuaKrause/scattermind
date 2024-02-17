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
"""Defines how a task response contains."""
from typing import cast, get_args, Literal, TYPE_CHECKING, TypedDict

from scattermind.system.logger.error import ErrorInfo, raise_error, warn_error
from scattermind.system.names import GNamespace


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
TASK_STATUS_ERROR: TaskStatus = "error"
"""An error occured while executing the task."""
TASK_STATUS_UNKNOWN: TaskStatus = "unknown"
"""The status of the task is unknown. The task might not exist."""


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
    "result": 'TaskValueContainer | None',
    "duration": float,
    "retries": int,
    "error": ErrorInfo | None,
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
