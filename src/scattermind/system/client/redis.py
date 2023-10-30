from collections.abc import Iterable
from typing import Literal, TypeVar

from redipy import Redis, RedisConfig
from redipy.api import PipelineAPI
from redipy.graph.expr import JSONType

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
from scattermind.system.logger.error import (
    ErrorInfo,
    from_error_json,
    to_error_json,
)
from scattermind.system.names import NName, QualifiedName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import DataContainer, TaskValueContainer
from scattermind.system.redis_util import (
    redis_to_robj,
    redis_to_tvc,
    robj_to_redis,
    RStack,
    tvc_to_redis,
)
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_INIT,
    TASK_STATUS_UNKNOWN,
    TaskStatus,
    to_status,
)
from scattermind.system.util import get_time_str


DT = TypeVar('DT', bound=DataId)


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
    "error",  # ErrorJSON str
]


class RedisClientPool(ClientPool):
    def __init__(self, cfg: RedisConfig) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="client")
        self._stack = RStack(self._redis)

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

    def delete(
            self,
            pipe: PipelineAPI,
            name: KeyName,
            task_id: TaskId) -> None:
        pipe.delete(self.key(name, task_id))

    def get_value(
            self,
            name: KeyName,
            task_id: TaskId) -> str | None:
        return self._redis.get(self.key(name, task_id))

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
            self._stack.init(self.key("stack_data", task_id), pipe=pipe)
            # NOTE: no need to set stack_frame yet
            pipe.execute()  # FIXME remove once on redipy 0.4.0
        return task_id

    def init_data(
            self,
            store: DataStore,
            task_id: TaskId,
            input_format: DataFormat) -> None:
        value = self.get_value("values", task_id)
        if value is None:
            raise ValueError(f"unknown task {task_id}")
        original_input = redis_to_tvc(value, input_format)
        byte_size = original_input.byte_size(input_format)
        stack_data: DataContainer = {}  # TODO: directly place instead
        original_input.place_data(None, store, input_format, stack_data)
        stack_key = self.key("stack_data", task_id)
        # TODO batch everything together
        for field, data_id in stack_data.items():
            self._stack.set_value(
                stack_key, field.to_parseable(), data_id.to_parseable())
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "byte_size", task_id, f"{byte_size}")
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def set_bulk_status(
            self,
            task_ids: Iterable[TaskId],
            status: TaskStatus) -> list[TaskId]:
        with self._redis.pipeline() as pipe:
            res = []
            for task_id in task_ids:
                self.set_value(pipe, "status", task_id, status)
                res.append(task_id)
            pipe.execute()  # FIXME remove once on redipy 0.4.0
            return res

    def get_status(self, task_id: TaskId) -> TaskStatus:
        res = self.get_value("status", task_id)
        if res is None:
            res = TASK_STATUS_UNKNOWN
        return to_status(res)

    def set_final_output(
            self, task_id: TaskId, final_output: TaskValueContainer) -> None:
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "result", task_id, tvc_to_redis(final_output))
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def get_final_output(
            self,
            task_id: TaskId,
            output_format: DataFormat) -> TaskValueContainer | None:
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "status", task_id, TASK_STATUS_DONE)
            # TODO create function to properly allow grouping
            pipe.get(self.key("result", task_id))
            _, res = pipe.execute()
        if res is None:
            return None
        return redis_to_tvc(res, output_format)

    def set_error(self, task_id: TaskId, error_info: ErrorInfo) -> None:
        with self._redis.pipeline() as pipe:
            self.set_value(
                pipe,
                "error",
                task_id,
                robj_to_redis(to_error_json(error_info)))
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def get_error(self, task_id: TaskId) -> ErrorInfo | None:
        res = self.get_value("error", task_id)
        if res is None:
            return None
        return from_error_json(redis_to_robj(res))

    def inc_retries(self, task_id: TaskId) -> int:
        return int(self._redis.incrby(self.key("retries", task_id), 1))

    def get_retries(self, task_id: TaskId) -> int:
        res = self.get_value("retries", task_id)
        if res is None:
            return 0
        return int(res)

    def get_task_start(self, task_id: TaskId) -> str:
        res = self.get_value("start_time", task_id)
        if res is None:
            raise ValueError(f"no start time set for {task_id}")
        return res

    def set_duration_value(self, task_id: TaskId, seconds: float) -> None:
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "duration", task_id, f"{seconds}")
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def get_duration(self, task_id: TaskId) -> float:
        res = self.get_value("duration", task_id)
        if res is None:
            raise ValueError(f"duration for {task_id} not set")
        return float(res)

    def commit_task(
            self,
            task_id: TaskId,
            data: DataContainer,
            *,
            weight: float,
            byte_size: int,
            push_frame: tuple[NName, GraphId, QueueId] | None) -> None:
        # FIXME proper operation grouping
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "weight", task_id, f"{weight}")
            pipe.incrby(self.key("byte_size", task_id), byte_size)
            stack_data_key = self.key("stack_data", task_id)
            if push_frame is not None:
                name, graph_id, qid = push_frame
                pipe.rpush(self.key("stack_frame", task_id), robj_to_redis({
                    "name": name.get(),
                    "graph_id": graph_id.to_parseable(),
                    "qid": qid.to_parseable(),
                }))
                self._stack.push_frame(stack_data_key)
            for field, data_id in data.items():
                self._stack.set_value(
                    stack_data_key,
                    field.to_parseable(),
                    data_id.to_parseable())
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def pop_frame(
            self,
            task_id: TaskId,
            data_id_type: type[DataId],
            ) -> tuple[tuple[NName, GraphId, QueueId] | None, DataContainer]:
        # FIXME proper operation grouping
        stack_data_key = self.key("stack_data", task_id)
        res = {
            QualifiedName.parse(field): data_id_type.parse(f"{value}")
            for field, value in self._stack.pop_frame(stack_data_key).items()
        }
        stack_frame_key = self.key("stack_frame", task_id)
        frame_res = self._redis.rpop(stack_frame_key)
        if frame_res is None:
            return None, res
        frame = redis_to_robj(frame_res)
        name = NName(frame["name"])
        graph_id = GraphId.parse(frame["graph_id"])
        qid = QueueId.parse(frame["qid"])
        return (name, graph_id, qid), res

    def get_weight(self, task_id: TaskId) -> float:
        res = self.get_value("weight", task_id)
        if res is None:
            raise ValueError(f"unknown task {task_id} for weight")
        return float(res)

    def get_byte_size(self, task_id: TaskId) -> int:
        res = self.get_value("byte_size", task_id)
        if res is None:
            raise ValueError(f"unknown task {task_id} for byte size")
        return int(res)

    def get_data(
            self,
            task_id: TaskId,
            vmap: ValueMap,
            data_id_type: type[DT]) -> dict[str, DT]:
        # FIXME proper operation grouping
        stack_data_key = self.key("stack_data", task_id)

        def as_data_id(qual: QualifiedName, text: JSONType | None) -> DT:
            if text is None:
                raise ValueError(f"no data id for {qual} in {task_id}")
            return data_id_type.parse(f"{text}")

        return {
            key: as_data_id(
                qual,
                self._stack.get_value(stack_data_key, qual.to_parseable()))
            for key, qual in vmap.items()
        }

    def clear_progress(self, task_id: TaskId) -> None:
        with self._redis.pipeline() as pipe:
            self.set_value(pipe, "weight", task_id, "1.0")
            self.set_value(pipe, "byte_size", task_id, "0")
            # FIXME might be buggy
            self.delete(pipe, "stack_data", task_id)
            pipe.execute()  # FIXME remove after bug is fixed
            self._stack.init(self.key("stack_data", task_id), pipe=pipe)
            self.delete(pipe, "stack_frame", task_id)
            pipe.execute()  # FIXME remove once on redipy 0.4.0

    def clear_task(self, task_id: TaskId) -> None:
        with self._redis.pipeline() as pipe:
            self.delete(pipe, "values", task_id)
            self.delete(pipe, "status", task_id)
            self.delete(pipe, "retries", task_id)
            self.delete(pipe, "result", task_id)
            self.delete(pipe, "start_time", task_id)
            self.delete(pipe, "duration", task_id)
            self.delete(pipe, "weight", task_id)
            self.delete(pipe, "byte_size", task_id)
            self.delete(pipe, "stack_data", task_id)
            self.delete(pipe, "stack_frame", task_id)
            self.delete(pipe, "error", task_id)
            pipe.execute()  # FIXME remove once on redipy 0.4.0
