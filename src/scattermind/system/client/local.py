import threading
from collections.abc import Iterable

from scattermind.system.base import DataId, GraphId, QueueId, TaskId
from scattermind.system.client.client import ClientPool
from scattermind.system.info import DataFormat
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.logger.error import ErrorInfo
from scattermind.system.names import NName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import DataContainer, TaskValueContainer
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_INIT,
    TASK_STATUS_UNKNOWN,
    TaskStatus,
)
from scattermind.system.util import get_time_str, seconds_since


class LocalClientPool(ClientPool):
    def __init__(self) -> None:
        super().__init__()
        self._values: dict[TaskId, TaskValueContainer] = {}
        self._status: dict[TaskId, TaskStatus] = {}
        self._retries: dict[TaskId, int] = {}
        self._start_times: dict[TaskId, str] = {}
        self._duration: dict[TaskId, float] = {}
        self._weight: dict[TaskId, float] = {}
        self._byte_size: dict[TaskId, int] = {}
        self._stack_data: dict[TaskId, list[DataContainer]] = {}
        self._stack_frame: dict[
            TaskId, list[tuple[NName, GraphId, QueueId]]] = {}
        self._results: dict[TaskId, TaskValueContainer | None] = {}
        self._error: dict[TaskId, ErrorInfo] = {}
        self._lock = threading.RLock()

    @staticmethod
    def is_local_only() -> bool:
        return True

    def create_task(
            self,
            original_input: TaskValueContainer) -> TaskId:
        with self._lock:
            task_id = TaskId.create()
            self._values[task_id] = original_input
            self._status[task_id] = TASK_STATUS_INIT
            self._retries[task_id] = 0
            self._results[task_id] = None
            self._start_times[task_id] = get_time_str()
            self._weight[task_id] = 1.0
            self._byte_size[task_id] = 0
            self._stack_data[task_id] = [{}]
            self._stack_frame[task_id] = []
            return task_id

    def init_data(
            self,
            store: DataStore,
            task_id: TaskId,
            input_format: DataFormat) -> None:
        with self._lock:
            stack_data = self._stack_data[task_id][-1]
            original_input = self._values[task_id]
            byte_size = original_input.byte_size(input_format)
            original_input.place_data(None, store, input_format, stack_data)
            self._byte_size[task_id] = byte_size
            print(f"{ctx_fmt()} placed {task_id} {self._stack_data[task_id]}")

    def set_bulk_status(
            self,
            task_ids: Iterable[TaskId],
            status: TaskStatus) -> list[TaskId]:
        with self._lock:
            res = []
            for task_id in task_ids:
                self._status[task_id] = status
                res.append(task_id)
            return res

    def get_status(self, task_id: TaskId) -> TaskStatus:
        return self._status.get(task_id, TASK_STATUS_UNKNOWN)

    def set_final_output(
            self, task_id: TaskId, final_output: TaskValueContainer) -> None:
        with self._lock:
            self._results[task_id] = final_output

    def get_final_output(self, task_id: TaskId) -> TaskValueContainer | None:
        with self._lock:
            res = self._results.get(task_id)
            self._status[task_id] = TASK_STATUS_DONE
            return res

    def set_error(self, task_id: TaskId, error_info: ErrorInfo) -> None:
        with self._lock:
            self._error[task_id] = error_info

    def get_error(self, task_id: TaskId) -> ErrorInfo | None:
        return self._error.get(task_id)

    def inc_retries(self, task_id: TaskId) -> int:
        with self._lock:
            self._retries[task_id] += 1
            return self._retries[task_id]

    def get_retries(self, task_id: TaskId) -> int:
        return self._retries[task_id]

    def get_task_start(self, task_id: TaskId) -> str:
        return self._start_times[task_id]

    def set_duration_value(self, task_id: TaskId, seconds: float) -> None:
        with self._lock:
            self._duration[task_id] = seconds

    def get_duration(self, task_id: TaskId) -> float:
        res = self._duration.get(task_id)
        if res is None:  # FIXME reevaluate if this is necessary
            return seconds_since(self.get_task_start(task_id))
        return res

    def commit_task(
            self,
            task_id: TaskId,
            data: DataContainer,
            *,
            weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None) -> None:
        with self._lock:
            print(f"{ctx_fmt()} commit {task_id} {self._stack_data[task_id]}")
            self._weight[task_id] = weight
            self._byte_size[task_id] += byte_size
            if push_frame is not None:
                self._stack_frame[task_id].append(push_frame)
                self._stack_data[task_id].append({})
            frame_data = self._stack_data[task_id][-1]
            for key, data_id in data.items():
                frame_data[key] = data_id
            print(f"{ctx_fmt()} frame_data {task_id} {frame_data}")

    def pop_frame(
            self,
            task_id: TaskId,
            ) -> tuple[tuple[NName, GraphId, QueueId] | None, DataContainer]:
        with self._lock:
            stack_frame = self._stack_frame[task_id]
            if stack_frame:
                res = stack_frame.pop()
            else:
                res = None
            stack_data = self._stack_data[task_id]
            print(f"{ctx_fmt()} pop {task_id}")
            return res, stack_data.pop()

    def get_weight(self, task_id: TaskId) -> float:
        return self._weight[task_id]

    def get_byte_size(self, task_id: TaskId) -> int:
        return self._byte_size[task_id]

    def get_data(self, task_id: TaskId, vmap: ValueMap) -> dict[str, DataId]:
        frame_data = self._stack_data[task_id][-1]
        print(f"{ctx_fmt()} get_data {task_id} {frame_data}")
        return {
            key: frame_data[qual]
            for key, qual in vmap.items()
        }

    def clear_progress(self, task_id: TaskId) -> None:
        with self._lock:
            self._weight[task_id] = 1.0
            self._byte_size[task_id] = 0
            self._stack_data[task_id] = [{}]
            self._stack_frame[task_id] = []
            print(f"{ctx_fmt()} clear progress {task_id}")

    def clear_task(self, task_id: TaskId) -> None:
        with self._lock:
            self._values.pop(task_id, None)
            self._status.pop(task_id, None)
            self._retries.pop(task_id, None)
            self._results.pop(task_id, None)
            self._start_times.pop(task_id, None)
            self._duration.pop(task_id, None)
            self._weight.pop(task_id, None)
            self._byte_size.pop(task_id, None)
            self._stack_data.pop(task_id, None)
            self._stack_frame.pop(task_id, None)
            self._error.pop(task_id, None)
            print(f"{ctx_fmt()} clear {task_id}")
