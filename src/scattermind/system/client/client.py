from collections.abc import Iterable
from typing import TYPE_CHECKING

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


TASK_MAX_RETRIES = 5


class ClientPool(Module):
    def set_duration(self, task_id: TaskId) -> None:
        self.set_duration_value(
            task_id, seconds_since(self.get_task_start(task_id)))

    def get_response(self, task_id: TaskId) -> ResponseObject:
        return {
            "status": self.get_status(task_id),
            "duration": self.get_duration(task_id),
            "retries": self.get_retries(task_id),
            "result": self.get_final_output(task_id),
            "error": self.get_error(task_id),
        }

    def create_task(self, original_input: 'TaskValueContainer') -> TaskId:
        raise NotImplementedError()

    def init_data(
            self,
            store: DataStore,
            task_id: TaskId,
            input_format: DataFormat) -> None:
        raise NotImplementedError()

    def set_bulk_status(
            self,
            task_ids: Iterable[TaskId],
            status: TaskStatus) -> list[TaskId]:
        raise NotImplementedError()

    def get_status(self, task_id: TaskId) -> TaskStatus:
        raise NotImplementedError()

    def set_final_output(
            self, task_id: TaskId, final_output: 'TaskValueContainer') -> None:
        raise NotImplementedError()

    def get_final_output(self, task_id: TaskId) -> 'TaskValueContainer | None':
        raise NotImplementedError()

    def set_error(self, task_id: TaskId, error_info: ErrorInfo) -> None:
        raise NotImplementedError()

    def get_error(self, task_id: TaskId) -> ErrorInfo | None:
        raise NotImplementedError()

    def inc_retries(self, task_id: TaskId) -> int:
        raise NotImplementedError()

    def get_retries(self, task_id: TaskId) -> int:
        raise NotImplementedError()

    def get_task_start(self, task_id: TaskId) -> str:
        raise NotImplementedError()

    def set_duration_value(self, task_id: TaskId, seconds: float) -> None:
        raise NotImplementedError()

    def get_duration(self, task_id: TaskId) -> float:
        raise NotImplementedError()

    def commit_task(
            self,
            task_id: TaskId,
            data: 'DataContainer',
            *,
            weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None) -> None:
        raise NotImplementedError()

    def pop_frame(
            self,
            task_id: TaskId,
            ) -> tuple[tuple[NName, GraphId, QueueId] | None, 'DataContainer']:
        raise NotImplementedError()

    def get_weight(self, task_id: TaskId) -> float:
        raise NotImplementedError()

    def get_byte_size(self, task_id: TaskId) -> int:
        raise NotImplementedError()

    def get_data(self, task_id: TaskId, vmap: ValueMap) -> dict[str, DataId]:
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

    def get_data_in(self) -> dict[str, DataId]:
        return self._cpool.get_data(self._task_id, self._vmap)

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
