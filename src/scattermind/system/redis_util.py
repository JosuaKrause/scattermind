import os
import threading
import uuid
from collections.abc import Mapping
from typing import cast, Literal, overload

import torch
from redipy import ExecFunction, RedisConfig
from redipy.api import PipelineAPI, RedisClientAPI
from redipy.graph.expr import JSONType
from redipy.symbolic.expr import Strs
from redipy.symbolic.fun import ToIntStr, ToNum
from redipy.symbolic.rhash import RedisHash
from redipy.symbolic.rvar import RedisVar
from redipy.symbolic.seq import FnContext

from scattermind.system.info import DataFormat
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.torch_util import (
    deserialize_tensor,
    DTypeName,
    serialize_tensor,
)
from scattermind.system.util import (
    as_base85,
    from_base85,
    json_compact,
    json_read,
)


TEST_SALT_LOCK = threading.RLock()
TEST_SALT: dict[str, str] = {}


def is_test() -> bool:
    test_id = os.getenv("PYTEST_CURRENT_TEST")
    return test_id is not None


def get_test_salt() -> str | None:
    test_id = os.getenv("PYTEST_CURRENT_TEST")
    if test_id is None:
        return None
    res = TEST_SALT.get(test_id)
    if res is None:
        with TEST_SALT_LOCK:
            res = TEST_SALT.get(test_id)
            if res is None:
                res = f"salt:{uuid.uuid4().hex}"
                TEST_SALT[test_id] = res
    return res


def get_test_config() -> RedisConfig:
    return {
        "host": "localhost",
        "port": 6380,
        "passwd": "",
        "prefix": f"test:{get_test_salt()}",
        "path": "userdata/test/",
    }


DataMode = Literal["size", "time"]
DM_SIZE: DataMode = "size"
DM_TIME: DataMode = "time"


def bytes_to_redis(value: bytes) -> str:
    return as_base85(value)


def redis_to_bytes(text: str) -> bytes:
    return from_base85(text)


@overload
def maybe_redis_to_bytes(text: str) -> bytes:
    ...


@overload
def maybe_redis_to_bytes(text: None) -> None:
    ...


def maybe_redis_to_bytes(text: str | None) -> bytes | None:
    if text is None:
        return None
    return redis_to_bytes(text)


def tensor_to_redis(value: torch.Tensor) -> str:
    return bytes_to_redis(serialize_tensor(value))


def redis_to_tensor(text: str, dtype: DTypeName) -> torch.Tensor:
    return deserialize_tensor(redis_to_bytes(text), dtype)


def robj_to_redis(obj: Mapping) -> str:
    return json_compact(obj)


def redis_to_robj(text: str) -> dict:
    return json_read(text)


TVCObj = dict[str, str]


def tvc_to_robj(tvc: TaskValueContainer) -> TVCObj:
    return {
        key: tensor_to_redis(value)
        for key, value in tvc.items()
    }


def robj_to_tvc(obj: TVCObj, value_format: DataFormat) -> TaskValueContainer:
    return TaskValueContainer({
        name: fmt.check_tensor(redis_to_tensor(obj[name], fmt.dtype()))
        for name, fmt in value_format.items()
    })


def tvc_to_redis(tvc: TaskValueContainer) -> str:
    return robj_to_redis(tvc_to_robj(tvc))


def redis_to_tvc(text: str, value_format: DataFormat) -> TaskValueContainer:
    return robj_to_tvc(redis_to_robj(text), value_format)


class RStack:
    def __init__(self, rt: RedisClientAPI) -> None:
        self._rt = rt

        self._set_value = self._set_value_script()
        self._get_value = self._get_value_script()
        self._pop_frame = self._pop_frame_script()
        self._get_cascading = self._get_cascading_script()

    def key(self, base: str, name: str) -> str:
        return f"{base}:{name}"

    def init(self, base: str, *, pipe: PipelineAPI | None) -> None:
        if pipe is None:
            self._rt.set(self.key(base, "size"), "0")
        else:
            pipe.set(self.key(base, "size"), "0")

    def push_frame(self, base: str) -> None:
        self._rt.incrby(self.key(base, "size"), 1)

    def pop_frame(self, base: str) -> dict[str, JSONType]:
        keys = {
            "size": self.key(base, "size"),
            "frame": self.key(base, "frame"),
        }
        args: dict[str, JSONType] = {}
        res = self._pop_frame(keys=keys, args=args)
        if isinstance(res, list):  # FIXME: remove in 0.4.0 -- redipy 0.3.0 bug
            obj: dict[str, JSONType] = {}
            key = None
            for val in res:
                if key is None:
                    key = val
                else:
                    obj[key] = val
                    key = None
            return obj
        return cast(dict, res)

    def set_value(self, base: str, field: str, value: str) -> None:
        keys = {
            "size": self.key(base, "size"),
            "frame": self.key(base, "frame"),
        }
        args: dict[str, JSONType] = {"field": field, "value": value}
        self._set_value(keys=keys, args=args)

    def get_value(
            self, base: str, field: str, *, cascade: bool = False) -> JSONType:
        keys = {
            "size": self.key(base, "size"),
            "frame": self.key(base, "frame"),
        }
        args: dict[str, JSONType] = {"field": field}
        if cascade:
            return self._get_cascading(keys=keys, args=args)
        return self._get_value(keys=keys, args=args)

    def _set_value_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        rframe = RedisHash(Strs(
            ctx.add_key("frame"),
            ":",
            ToIntStr(rsize.get(no_adjust=True).or_(0))))
        field = ctx.add_arg("field")
        value = ctx.add_arg("value")
        ctx.add(rframe.hset({
            field: value,
        }))
        ctx.set_return_value(None)
        return self._rt.register_script(ctx)

    def _get_value_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        rframe = RedisHash(Strs(
            ctx.add_key("frame"),
            ":",
            ToIntStr(rsize.get(no_adjust=True).or_(0))))
        field = ctx.add_arg("field")
        ctx.set_return_value(rframe.hget(field))
        return self._rt.register_script(ctx)

    def _pop_frame_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        rframe = RedisHash(Strs(
            ctx.add_key("frame"),
            ":",
            ToIntStr(rsize.get(no_adjust=True).or_(0))))
        res = ctx.add_local(rframe.hgetall())
        ctx.add(rframe.delete())
        ctx.add(rsize.incrby(-1))
        ctx.set_return_value(res)
        return self._rt.register_script(ctx)

    def _get_cascading_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        base = ctx.add_local(ctx.add_key("frame"))
        field = ctx.add_arg("field")
        pos = ctx.add_local(ToNum(rsize.get(no_adjust=True).or_(0)))
        res = ctx.add_local(None)
        cur = ctx.add_local(None)
        rframe = RedisHash(cur)

        loop = ctx.while_(res.eq_(None).and_(pos.ge_(0)))
        loop.add(cur.assign(Strs(base, ":", ToIntStr(pos))))
        loop.add(res.assign(rframe.hget(field)))
        loop.add(pos.assign(pos - 1))

        ctx.set_return_value(res)
        return self._rt.register_script(ctx)
