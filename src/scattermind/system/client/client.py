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
from scattermind.system.names import GNamespace, NName, ValueMap
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
            output_format: DataFormat | None) -> ResponseObject:
        """
        Retrieves the summary of the task. If the final output are available or
        the task caused an error, the respective fields are set.

        Args:
            task_id (TaskId): The task id.
            output_format (DataFormat | None): The expected data format of the
                final output. If the final output format is not known (e.g.,
                if the namespace is not available) and the value is None the
                result field will automatically be None as well.

        Returns:
            ResponseObject: The task summary.
        """
        return {
            "ns": self.get_namespace(task_id),
            "status": self.get_status(task_id),
            "duration": self.get_duration(task_id),
            "retries": self.get_retries(task_id),
            "result":
                None
                if output_format is None
                else self.get_final_output(task_id, output_format),
            "error": self.get_error(task_id),
        }

    def create_task(
            self,
            ns: GNamespace,
            original_input: 'TaskValueContainer') -> TaskId:
        """
        Create a new task from a given input.

        Args:
            ns (GNamespace): The namespace.
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

    def get_namespace(self, task_id: TaskId) -> GNamespace | None:
        """
        Retrieves the namespace of the given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            GNamespace | None: The namespace or None if the task does not
                exist.
        """
        raise NotImplementedError()

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """
        Retrieves the status of the given task.

        Args:
            task_id (TaskId): The task id.

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
            push_frame: tuple[NName, GraphId, QueueId] | None) -> GNamespace:
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
                the calling node (the node initiating the push; this is used
                later to make the results of the subgraph available in the
                calling graph as value fields of the caller node), the new
                graph id (the graph that gets called), and the return queue
                after the (sub-)graph computation has completed.
        """
        raise NotImplementedError()

    def pop_frame(
            self,
            task_id: TaskId,
            data_id_type: type[DataId],
            ) -> tuple[tuple[NName, GraphId, QueueId] | None, 'DataContainer']:
        """
        Pops the current frame from the stack. This function should be called
        when computing a subgraph and obtaining the final result.

        Args:
            task_id (TaskId): The task id.
            data_id_type (type[DataId]): The data id subclass compatible with
                the payload data store.

        Returns:
            tuple[tuple[NName, GraphId, QueueId] | None, DataContainer]:
                Returns the calling tuple and the full stack frame. The calling
                tuple consists of the caller node name (to make the results
                of the subgraph available in the caller graph), the graph id
                of the subgraph, and the return queue id that defines where to
                continue execution in the caller graph after the subgraph has
                completed.
        """
        raise NotImplementedError()

    def get_weight(self, task_id: TaskId) -> float:
        """
        Retrieve the current weight of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            float: The weight.
        """
        raise NotImplementedError()

    def get_byte_size(self, task_id: TaskId) -> int:
        """
        Retrieve the current byte size of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            int: The byte size.
        """
        raise NotImplementedError()

    def get_data(
            self,
            task_id: TaskId,
            vmap: ValueMap,
            data_id_type: type[DT]) -> dict[str, DT]:
        """
        Load all data specified by the value map from the current stack frame.

        Args:
            task_id (TaskId): The task id.
            vmap (ValueMap): The value map defining which data to load.
            data_id_type (type[DT]): The data id compatible with the payload
                data store.

        Returns:
            dict[str, DT]: The data ids for each input field name keys.
        """
        raise NotImplementedError()

    def clear_progress(self, task_id: TaskId) -> None:
        """
        Reset the progress of a task.

        Args:
            task_id (TaskId): The task id.
        """
        raise NotImplementedError()

    def clear_task(self, task_id: TaskId) -> None:
        """
        Remove all data associated with a task.

        Args:
            task_id (TaskId): The task id.
        """
        raise NotImplementedError()

    @staticmethod
    def get_max_retries() -> int:
        """
        Get the maximum number of times a task can attempt to be executed.

        Returns:
            int: The maximum number of allowed retries.
        """
        return TASK_MAX_RETRIES


class ComputeTask:
    """A local representation / state of a task. Changes to this wrapper are
    only visible locally (by the current executor) until the changes are
    committed to the client pool. Only data relevant to the current execution
    is loaded.
    """
    def __init__(
            self,
            cpool: ClientPool,
            task_id: TaskId,
            vmap: ValueMap) -> None:
        """
        Creates a compute task.

        Args:
            cpool (ClientPool): The client pool.
            task_id (TaskId): The task id.
            vmap (ValueMap): Which data values to make available for the
                current computation.
        """
        self._cpool = cpool
        self._task_id = task_id
        self._vmap = vmap
        self._weight_out: float | None = None
        self._byte_size_out: int | None = None
        self._push_frame: tuple[NName, GraphId, QueueId] | None = None
        self._data_out: 'DataContainer | None' = None
        self._next_qid: QueueId | None = None

    def get_task_id(self) -> TaskId:
        """
        Get the task id.

        Returns:
            TaskId: The task id.
        """
        return self._task_id

    def get_simple_weight_in(self) -> float:
        """
        Computes the simple weight of the task.

        Returns:
            float: The task weight.
        """
        return self._cpool.get_weight(self._task_id)

    def get_total_weight_in(self) -> float:
        """
        Computes the pressure created by this task.

        Returns:
            float: The total pressure of this task.
        """
        return self.get_simple_weight_in() * self.get_byte_size_in()

    def get_byte_size_in(self) -> int:
        """
        The byte size of the task.

        Returns:
            int: The byte size.
        """
        return self._cpool.get_byte_size(self._task_id)

    def get_data_in(self, data_id_type: type[DT]) -> dict[str, DT]:
        """
        Get the input data for the current execution.

        Args:
            data_id_type (type[DT]): The data id type compatible with the
                payload data store.

        Returns:
            dict[str, DT]: The data needed for the current execution.
        """
        return self._cpool.get_data(self._task_id, self._vmap, data_id_type)

    def get_data_out(self) -> 'DataContainer':
        """
        The current output data.

        Raises:
            ValueError: If the output data has not been set.

        Returns:
            DataContainer: The current output data.
        """
        if self._data_out is None:
            raise ValueError("no output data set")
        return self._data_out

    def get_weight_out(self) -> float:
        """
        The weight of the task on the next node.

        Raises:
            ValueError: If the next weight has not been set.

        Returns:
            float: The weight.
        """
        if self._weight_out is None:
            raise ValueError("no output weight set")
        return self._weight_out

    def get_byte_size_out(self) -> int:
        """
        The byte size of the task for the next node.

        Raises:
            ValueError: If the next size has not been set.

        Returns:
            int: The byte size.
        """
        if self._byte_size_out is None:
            raise ValueError("no output byte size set")
        return self._byte_size_out

    def get_push_frame(self) -> tuple[NName, GraphId, QueueId] | None:
        """
        Get the push frame info if it has been set.

        Returns:
            tuple[NName, GraphId, QueueId] | None: If set, the tuple is the
                caller node name from the calling graph. This is used for
                making the results of the subgraph available in the calling
                graph. The second element of the tuple is the subgraph id to
                identify the entry point for the subgraph execution. The third
                element of the tuple is the return queue id to identify where
                to continue execution in the calling graph after the subgraph
                execution has finished.
        """
        return self._push_frame

    def get_next_queue_id(self) -> QueueId:
        """
        Get the next queue id to continue execution in.

        Raises:
            ValueError: If the next queue id has not been set.

        Returns:
            QueueId: The queue id.
        """
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
        """
        Set the result of the current execution.

        Args:
            data_out (DataContainer): The result of the current execution.
            add_weight (float): How much weight to add to the task.
            byte_size (int): The newly added byte size to the task.
            push_frame (tuple[NName, GraphId, QueueId] | None): The push frame
                if a subgraph calculation is triggered. The tuple consists of
                the calling node name (for providing the results of the
                subgraph to the calling graph), the subgraph id (defining the
                entry point of the subgraph calculation), and the return queue
                id to continue computation after the subgraph computation has
                finished.
            next_qid (QueueId): The queue id of the queue where the task is
                added next for execution.

        Raises:
            ValueError: If the results have already been set.
        """
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
        """
        Whether the result of the current computation have already been set.

        Returns:
            bool: True if the results have been set.
        """
        return self._data_out is not None

    @staticmethod
    def get_total_byte_size(tasks: list['ComputeTask']) -> int:
        """
        Compute the total byte size of a list of tasks.

        Args:
            tasks (list[ComputeTask]): The list of tasks.

        Returns:
            int: The total byte size of the list of tasks.
        """
        return sum(task.get_byte_size_in() for task in tasks)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}["
            f"task_id={self._task_id},"
            f"has_result={self.has_result()}]")

    def __repr__(self) -> str:
        return self.__str__()
