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
"""A RAM-only client pool."""
import threading
from collections.abc import Iterable
from typing import TypeVar

from scattermind.system.base import (
    DataId,
    GraphId,
    L_LOCAL,
    Locality,
    QueueId,
    TaskId,
)
from scattermind.system.client.client import ClientPool
from scattermind.system.info import DataFormat
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.logger.error import ErrorInfo
from scattermind.system.names import GNamespace, NName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import DataContainer, TaskValueContainer
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_INIT,
    TASK_STATUS_UNKNOWN,
    TaskStatus,
)
from scattermind.system.util import get_time_str, seconds_since


DT = TypeVar('DT', bound=DataId)
"""The `DataId` subclass understood by a given `DataStore` implementation."""


class LocalClientPool(ClientPool):
    """A RAM-only client pool."""
    def __init__(self) -> None:
        """
        Creates a RAM-only client pool.
        """
        super().__init__()
        self._namespaces: dict[TaskId, GNamespace] = {}
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
    def locality() -> Locality:
        return L_LOCAL

    def create_task(
            self,
            ns: GNamespace,
            original_input: TaskValueContainer) -> TaskId:
        with self._lock:
            task_id = TaskId.create()
            self._namespaces[task_id] = ns
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

    def get_namespace(self, task_id: TaskId) -> GNamespace | None:
        return self._namespaces.get(task_id)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        return self._status.get(task_id, TASK_STATUS_UNKNOWN)

    def set_final_output(
            self, task_id: TaskId, final_output: TaskValueContainer) -> None:
        with self._lock:
            self._results[task_id] = final_output

    def get_final_output(
            self,
            task_id: TaskId,
            output_format: DataFormat) -> TaskValueContainer | None:
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
        if res is None:
            return seconds_since(self.get_task_start(task_id))
        return res

    def commit_task(
            self,
            task_id: TaskId,
            data: DataContainer,
            *,
            weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None) -> GNamespace:
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
            return self._namespaces[task_id]

    def pop_frame(
            self,
            task_id: TaskId,
            data_id_type: type[DataId],
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

    def get_data(
            self,
            task_id: TaskId,
            vmap: ValueMap,
            data_id_type: type[DT]) -> dict[str, DT]:
        frame_data = self._stack_data[task_id][-1]
        print(f"{ctx_fmt()} get_data {task_id} {frame_data}")

        def ensure_id(data_id: DataId) -> DT:
            if not isinstance(data_id, data_id_type):
                raise TypeError(
                    f"invalid data id: {data_id} expected type {data_id_type}")
            return data_id

        return {
            key: ensure_id(frame_data[qual])
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
            self._namespaces.pop(task_id, None)
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
