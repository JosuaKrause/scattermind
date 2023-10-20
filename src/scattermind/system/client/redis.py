from collections.abc import Iterable
from typing import Literal

from redipy import Redis, RedisConfig
from redipy.api import PipelineAPI

from scattermind.system.base import (
    DataId,
    GraphId,
    L_REMOTE,
    Locality,
    QueueId,
    TaskId,
)
from scattermind.system.client.client import ClientPool
from scattermind.system.info import DataFormat
from scattermind.system.logger.error import ErrorInfo
from scattermind.system.names import NName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import DataContainer, TaskValueContainer
from scattermind.system.redis_util import robj_to_redis, tvc_to_redis
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_INIT,
    TASK_STATUS_UNKNOWN,
    TaskStatus,
)
from scattermind.system.util import get_time_str


KeyName = Literal[
    "values",  # TVC str
    "status",  # str
    "retries",  # int
    "start_time",  # time str
    "duration",  # float
    "weight",  # float
    "byte_size",  # int
    "stack_data",  # list obj str
    "stack_frame",  # list obj str
    "result",  # TVC str
    "error",  # ErrorInfo str
]


class LocalClientPool(ClientPool):
    def __init__(self, cfg: RedisConfig) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg)

    @staticmethod
    def locality() -> Locality:
        return L_REMOTE

    @staticmethod
    def key(name: KeyName, task_id: TaskId) -> str:
        return f"{name}:{task_id.to_parseable()}"

    def set_value(
            self,
            pipe: PipelineAPI,
            name: KeyName,
            task_id: TaskId,
            value: str) -> None:
        pipe.set(self.key(name, task_id), value)

    def get_value(
            self,
            name: KeyName,
            task_id: TaskId) -> str | None:
        return self._redis.get(self.key(name, task_id))

    def push_value(
            self,
            pipe: PipelineAPI,
            name: KeyName,
            task_id: TaskId,
            value: str) -> None:
        pipe.rpush(self.key(name, task_id), value)

    # def peek_value(self, name: KeyName, task_id: TaskId) -> str:
    #     return self._redis

    def pop_value(self, name: KeyName, task_id: TaskId) -> str:
        res = self._redis.rpop(self.key(name, task_id))
        if res is None:
            raise KeyError(f"no {task_id} for {name}")
        return res

    def create_task(
            self,
            original_input: TaskValueContainer) -> TaskId:
        task_id = TaskId.create()
        with self._redis.pipeline() as pipe:
            self.set_value(
                pipe, "values", task_id, tvc_to_redis(original_input))
            self.set_value(pipe, "status", task_id, TASK_STATUS_INIT)
            self.set_value(pipe, "retries", task_id, f"{0}")
            # NOTE: no need to set result yet
            self.set_value(pipe, "start_time", task_id, get_time_str())
            self.set_value(pipe, "weight", task_id, f"{1.0}")
            self.set_value(pipe, "byte_size", task_id, f"{0}")
            self.push_value(pipe, "stack_data", task_id, robj_to_redis({}))
            # NOTE: no need to set stack_frame
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
