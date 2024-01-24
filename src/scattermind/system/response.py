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
from typing import cast, get_args, Literal, TYPE_CHECKING, TypedDict

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
TASK_STATUS_ALL: set[TaskStatus] = set(get_args(TaskStatus))
TASK_STATUS_INIT: TaskStatus = "init"
TASK_STATUS_WAIT: TaskStatus = "wait"
TASK_STATUS_BUSY: TaskStatus = "busy"
TASK_STATUS_READY: TaskStatus = "ready"
TASK_STATUS_DONE: TaskStatus = "done"
TASK_STATUS_ERROR: TaskStatus = "error"
TASK_STATUS_UNKNOWN: TaskStatus = "unknown"


def to_status(text: str) -> TaskStatus:
    if text not in TASK_STATUS_ALL:
        raise ValueError(f"invalid status {text}")
    return cast(TaskStatus, text)


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
