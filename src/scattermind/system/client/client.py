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
"""Provides the client pool interface."""
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

from scattermind.system.base import DataId, GraphId, Module, QueueId, TaskId
from scattermind.system.info import DataFormat
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.logger.error import ErrorInfo
from scattermind.system.names import NName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.response import ResponseObject, TaskStatus
from scattermind.system.util import seconds_since


if TYPE_CHECKING:
    from scattermind.system.payload.values import (
        DataContainer,
        TaskValueContainer,
    )


DT = TypeVar('DT', bound=DataId)
"""The `DataId` subclass understood by a given `DataStore` implementation."""


TASK_MAX_RETRIES = 5
"""The maximum number the same task can be repeated for executed before giving
up and setting the error and updating the state to error."""


class ClientPool(Module):
    """A client pool manages tasks. All data needed to start a task and all
    data read from completed tasks are stored in the client pool."""
    def set_duration(self, task_id: TaskId) -> None:
        """
        Sets the execution time of the task.

        Args:
            task_id (TaskId): The task id.
        """
        self.set_duration_value(
            task_id, seconds_since(self.get_task_start(task_id)))

    def get_response(
            self,
            task_id: TaskId,
            output_format: DataFormat) -> ResponseObject:
        """
        Retrieves the summary of the task. If the final output are available or
        the task caused an error, the respective fields are set.

        Args:
            task_id (TaskId): The task id.
            output_format (DataFormat): The expected data format of the final
                output.

        Returns:
            ResponseObject: The task summary.
        """
        return {
            "status": self.get_status(task_id),
            "duration": self.get_duration(task_id),
            "retries": self.get_retries(task_id),
            "result": self.get_final_output(task_id, output_format),
            "error": self.get_error(task_id),
        }

    def create_task(self, original_input: 'TaskValueContainer') -> TaskId:
        """
        Create a new task from a given input.

        Args:
            original_input (TaskValueContainer): The input data for the task.

        Returns:
            TaskId: The new task id.
        """
        raise NotImplementedError()

    def init_data(
            self,
            store: DataStore,
            task_id: TaskId,
            input_format: DataFormat) -> None:
        """
        Initializes payload data for execution of a task. Note, the input data
        of a task is always stored in the client pool. Here, we are moving that
        data to the (volatile) payload data store so the task can be executed.

        Args:
            store (DataStore): The payload data store.
            task_id (TaskId): The task id.
            input_format (DataFormat): The expected input format of the graph.
        """
        raise NotImplementedError()

    def set_bulk_status(
            self,
            task_ids: Iterable[TaskId],
            status: TaskStatus) -> list[TaskId]:
        """
        Sets the status for multiple tasks.

        Args:
            task_ids (Iterable[TaskId]): All affected tasks.
            status (TaskStatus): The new status.

        Returns:
            list[TaskId]: The list of tasks that got affected.
        """
        raise NotImplementedError()

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """
        Retrieves the status of the given task.

        Args:
            task_id (TaskId): The task id

        Returns:
            TaskStatus: The status.
        """
        raise NotImplementedError()

    def set_final_output(
            self, task_id: TaskId, final_output: 'TaskValueContainer') -> None:
        """
        Sets the final output of the task. Note, after this function completes
        successfully, the final output is stored in the client pool (instead
        of the volatile payload data storage) and is guaranteed to be
        available. However, after reading the results and returning the task
        summary all information of the task can be freed if needed.

        Args:
            task_id (TaskId): The task id.
            final_output (TaskValueContainer): The final output of the graph.
        """
        raise NotImplementedError()

    def get_final_output(
            self,
            task_id: TaskId,
            output_format: DataFormat) -> 'TaskValueContainer | None':
        """
        Retrieves the final output of the graph.

        Args:
            task_id (TaskId): The task id.
            output_format (DataFormat): The expected format of the final
                output.

        Returns:
            TaskValueContainer | None: The final output or None if the output
                has not been set.
        """
        raise NotImplementedError()

    def set_error(self, task_id: TaskId, error_info: ErrorInfo) -> None:
        """
        Sets the error value of the task. This does not imply that the task
        becomes inactive. A task can attempt to compute again even after it
        encountered an error.

        Args:
            task_id (TaskId): The task id.
            error_info (ErrorInfo): The error.
        """
        raise NotImplementedError()

    def get_error(self, task_id: TaskId) -> ErrorInfo | None:
        """
        Retrieves the last error encountered during task execution. Note, if
        an error is set it does not imply that the task has failed. Use the
        status for determining that. Even if the task completed successfully
        a previous execution attempt might have encountered an error.

        Args:
            task_id (TaskId): The task id.

        Returns:
            ErrorInfo | None: The error or None if no error occurred.
        """
        raise NotImplementedError()

    def inc_retries(self, task_id: TaskId) -> int:
        """
        Increase the number of retries of the task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            int: The new retries number.
        """
        raise NotImplementedError()

    def get_retries(self, task_id: TaskId) -> int:
        """
        Retrieves the number of times this task has been restarted.

        Args:
            task_id (TaskId): The task id.

        Returns:
            int: The number of retries. If this is the first execution attempt,
                the number is 0.
        """
        raise NotImplementedError()

    def get_task_start(self, task_id: TaskId) -> str:
        """
        Returns an ISO formatted time string that indicates the start time of
        the task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            str: When the task was created as ISO formatted time string.
        """
        raise NotImplementedError()

    def set_duration_value(self, task_id: TaskId, seconds: float) -> None:
        """
        Sets the total execution time of the task in seconds. The total
        execution time includes the time from task creation until the final
        result was set.

        Args:
            task_id (TaskId): The task id.
            seconds (float): The execution time in seconds.
        """
        raise NotImplementedError()

    def get_duration(self, task_id: TaskId) -> float:
        """
        Retrieves the total execution time of the task in seconds. The total
        execution time includes the time from task creation until the final
        result was set. If the task has not completed yet the current execution
        time is returned.

        Args:
            task_id (TaskId): The task id.

        Returns:
            float: The execution time in seconds.
        """
        raise NotImplementedError()

    def commit_task(
            self,
            task_id: TaskId,
            data: 'DataContainer',
            *,
            weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None) -> None:
        """
        Commit the current state of the task. This updates the state of the
        task and makes the new state visible to other executors.

        Args:
            task_id (TaskId): The task id.
            data (DataContainer): The data of the current stack frame.
            weight (float): The current weight of the task.
            byte_size (int): The current size in bytes of the task.
            push_frame (tuple[NName, GraphId, QueueId] | None): If non-None,
                a new stack frame is created. The tuple indicates the name of
                the calling node (the node initiating the push), the new graph
                id (the graph that gets called), and the return queue after the
                (sub-)graph computation is completed.
        """
        raise NotImplementedError()

    def pop_frame(
            self,
            task_id: TaskId,
            data_id_type: type[DataId],
            ) -> tuple[tuple[NName, GraphId, QueueId] | None, 'DataContainer']:
        raise NotImplementedError()

    def get_weight(self, task_id: TaskId) -> float:
        raise NotImplementedError()

    def get_byte_size(self, task_id: TaskId) -> int:
        raise NotImplementedError()

    def get_data(
            self,
            task_id: TaskId,
            vmap: ValueMap,
            data_id_type: type[DT]) -> dict[str, DT]:
        raise NotImplementedError()

    def clear_progress(self, task_id: TaskId) -> None:
        raise NotImplementedError()

    def clear_task(self, task_id: TaskId) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_max_retries() -> int:
        return TASK_MAX_RETRIES


class ComputeTask:
    def __init__(
            self,
            cpool: ClientPool,
            task_id: TaskId,
            vmap: ValueMap) -> None:
        self._cpool = cpool
        self._task_id = task_id
        self._vmap = vmap
        self._weight_out: float | None = None
        self._byte_size_out: int | None = None
        self._push_frame: tuple[NName, GraphId, QueueId] | None = None
        self._data_out: 'DataContainer | None' = None
        self._next_qid: QueueId | None = None

    def get_task_id(self) -> TaskId:
        return self._task_id

    def get_simple_weight_in(self) -> float:
        return self._cpool.get_weight(self._task_id)

    def get_total_weight_in(self) -> float:
        return self.get_simple_weight_in() * self.get_byte_size_in()

    def get_byte_size_in(self) -> int:
        return self._cpool.get_byte_size(self._task_id)

    def get_data_in(self, data_id_type: type[DT]) -> dict[str, DT]:
        return self._cpool.get_data(self._task_id, self._vmap, data_id_type)

    def get_data_out(self) -> 'DataContainer':
        if self._data_out is None:
            raise ValueError("no output data set")
        return self._data_out

    def get_weight_out(self) -> float:
        if self._weight_out is None:
            raise ValueError("no output weight set")
        return self._weight_out

    def get_byte_size_out(self) -> int:
        if self._byte_size_out is None:
            raise ValueError("no output byte size set")
        return self._byte_size_out

    def get_push_frame(self) -> tuple[NName, GraphId, QueueId] | None:
        return self._push_frame

    def get_next_queue_id(self) -> QueueId:
        if self._next_qid is None:
            raise ValueError("no next queue id set")
        return self._next_qid

    def set_result(
            self,
            data_out: 'DataContainer',
            add_weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None,
            next_qid: QueueId) -> None:
        if self.has_result():
            raise ValueError("result already set")
        assert add_weight > 0.0
        self._data_out = data_out
        self._weight_out = self.get_simple_weight_in() + add_weight
        self._byte_size_out = byte_size
        self._push_frame = push_frame
        self._next_qid = next_qid
        print(f"{ctx_fmt()} set result {self._task_id} {self._next_qid}")

    def has_result(self) -> bool:
        return self._data_out is not None

    @staticmethod
    def get_total_byte_size(tasks: list['ComputeTask']) -> int:
        return sum(task.get_byte_size_in() for task in tasks)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}["
            f"task_id={self._task_id},"
            f"has_result={self.has_result()}]")

    def __repr__(self) -> str:
        return self.__str__()
