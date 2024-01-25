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
"""Utility functions for redis."""
import os
import threading
import uuid
from collections.abc import Mapping
from typing import cast, Literal, overload

import torch
from redipy import ExecFunction, RedisConfig
from redipy.api import RedisClientAPI
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
"""Lock to ensure each test using redis gets a different salt for keys."""
TEST_SALT: dict[str, str] = {}
"""A dictionary of test salts by test name."""


def is_test() -> bool:
    """
    Whether we are currently running a pytest test.

    Returns:
        bool: True if running a pytest test.
    """
    test_id = os.getenv("PYTEST_CURRENT_TEST")
    return test_id is not None


def get_test_salt() -> str | None:
    """
    Returns the redis salt for the current test. The salt is to be used as
    prefix for all keys used by test case.

    Returns:
        str | None: The salt or None if we are not running a test.
    """
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
    """
    Get the redis connection details for test cases.

    Returns:
        RedisConfig: The redis connection details.
    """
    return {
        "host": "localhost",
        "port": 6380,
        "passwd": "",
        "prefix": f"test:{get_test_salt()}",
        "path": "userdata/test/",
    }


DataMode = Literal["size", "time"]
"""Whether the redis cache is limited by space or via expiration."""
DM_SIZE: DataMode = "size"
"""Indicates that the redis cache is limited in size."""
DM_TIME: DataMode = "time"
"""Indicates that the redis cache is limited by expiring keys after a set
time."""


def bytes_to_redis(value: bytes) -> str:
    """
    Convert a byte sequence to a value that can be used as redis value.

    Args:
        value (bytes): The byte sequence.

    Returns:
        str: The converted value.
    """
    return as_base85(value)


def redis_to_bytes(text: str) -> bytes:
    """
    Convert a previously encoded redis value back into a byte sequence.

    Args:
        text (str): The value from redis.

    Returns:
        bytes: The byte sequence.
    """
    return from_base85(text)


@overload
def maybe_redis_to_bytes(text: str) -> bytes:
    ...


@overload
def maybe_redis_to_bytes(text: None) -> None:
    ...


def maybe_redis_to_bytes(text: str | None) -> bytes | None:
    """
    Convert a previously encoded redis value back into a byte sequence.
    This function allows None as input and returns None in this case.

    Args:
        text (str | None): The value from redis or None.

    Returns:
        bytes: The byte sequence or None if the value was None.
    """
    if text is None:
        return None
    return redis_to_bytes(text)


def tensor_to_redis(value: torch.Tensor) -> str:
    """
    Convert a tensor into a string that can be saved on redis.

    Args:
        value (torch.Tensor): The tensor.

    Returns:
        str: The string.
    """
    return bytes_to_redis(serialize_tensor(value))


def redis_to_tensor(text: str, dtype: DTypeName) -> torch.Tensor:
    """
    Convert a previously encoded redis value back into a tensor.

    Args:
        text (str): The redis value.
        dtype (DTypeName): The expected dtype.

    Returns:
        torch.Tensor: The tensor.
    """
    return deserialize_tensor(redis_to_bytes(text), dtype)


def robj_to_redis(obj: Mapping) -> str:
    """
    Convert a JSON serializable object into a redis value.

    Args:
        obj (Mapping): The JSON serializable object.

    Returns:
        str: The redis value.
    """
    return json_compact(obj)


def redis_to_robj(text: str) -> dict:
    """
    Convert a redis value back into the JSON serializable object.

    Args:
        text (str): The redis value.

    Returns:
        dict: The JSON serializable object.
    """
    return json_read(text)


TVCObj = dict[str, str]
"""A serializable `TaskValueContainer` where values are converted to redis
compatible strings."""


def tvc_to_robj(tvc: TaskValueContainer) -> TVCObj:
    """
    Convert a task value container into a dictionary that can be encoded as
    JSON. The tensors are converted using ::py:function:`tensor_to_redis`.

    Args:
        tvc (TaskValueContainer): The task value container.

    Returns:
        TVCObj: A JSON serializable object.
    """
    return {
        key: tensor_to_redis(value)
        for key, value in tvc.items()
    }


def robj_to_tvc(obj: TVCObj, value_format: DataFormat) -> TaskValueContainer:
    """
    Convert a JSON serializable object back into a task value container.

    Args:
        obj (TVCObj): The JSON serializable object.
        value_format (DataFormat): The expected format of the task value
            container.

    Returns:
        TaskValueContainer: The task value container.
    """
    return TaskValueContainer({
        name: fmt.check_tensor(redis_to_tensor(obj[name], fmt.dtype()))
        for name, fmt in value_format.items()
    })


def tvc_to_redis(tvc: TaskValueContainer) -> str:
    """
    Convert a task value container into a redis value string. The tensors
    are converted using ::py:function:`tensor_to_redis`.

    Args:
        tvc (TaskValueContainer): The task value container.

    Returns:
        TVCObj: A redis value string.
    """
    return robj_to_redis(tvc_to_robj(tvc))


def redis_to_tvc(text: str, value_format: DataFormat) -> TaskValueContainer:
    """
    Convert a redis value back into a task value container.

    Args:
        text (str): The redis value.
        value_format (DataFormat): The expected format of the task value
            container.

    Returns:
        TaskValueContainer: The task value container.
    """
    return robj_to_tvc(redis_to_robj(text), value_format)


class RStack:
    """
    A dictionary stack in redis. Keys can be shadowed in stack frames.
    """
    def __init__(self, rt: RedisClientAPI) -> None:
        """
        Creates a dictionary stack for the given redis client.

        Args:
            rt (RedisClientAPI): The redis client.
        """
        self._rt = rt

        self._set_value = self._set_value_script()
        self._get_value = self._get_value_script()
        self._pop_frame = self._pop_frame_script()
        self._get_cascading = self._get_cascading_script()

    def key(self, base: str, name: str) -> str:
        """
        Compute the key.

        Args:
            base (str): The base key.

            name (str): The name.

        Returns:
            str: The key associated with the name.
        """
        return f"{base}:{name}"

    def push_frame(self, base: str) -> None:
        """
        Pushes a new stack frame.

        Args:
            base (str): The base key.
        """
        self._rt.incrby(self.key(base, "size"), 1)

    def pop_frame(self, base: str) -> dict[str, str]:
        """
        Pops the current stack frame and returns its values.

        Args:
            base (str): The base key.

        Returns:
            dict[str, str] | None: The content of the stack frame.
        """
        res = self._pop_frame(
            keys={
                "size": self.key(base, "size"),
                "frame": self.key(base, "frame"),
            },
            args={})
        if res is None:
            return {}
        return cast(dict, res)

    def set_value(self, base: str, field: str, value: str) -> None:
        """
        Set a value in the current stack frame.

        Args:
            base (str): The base key.

            field (str): The field.

            value (str): The value.
        """
        self._set_value(
            keys={
                "size": self.key(base, "size"),
                "frame": self.key(base, "frame"),
            },
            args={"field": field, "value": value})

    def get_value(
            self, base: str, field: str, *, cascade: bool = False) -> JSONType:
        """
        Returns a value from the stack. If the value is not in the current
        stack frame and cascade is set, the value is recursively retrieved
        from the previous stack frames.

        Args:
            base (str): The base key.

            field (str): The field.

            cascade (bool): Whether to recursively inspect all stack frames.

        Returns:
            JSONType: The value.
        """
        if cascade:
            return self._get_cascading(
                keys={
                    "size": self.key(base, "size"),
                    "frame": self.key(base, "frame"),
                },
                args={"field": field})
        return self._get_value(
            keys={
                "size": self.key(base, "size"),
                "frame": self.key(base, "frame"),
            },
            args={"field": field})

    def _set_value_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        rframe = RedisHash(Strs(
            ctx.add_key("frame"),
            ":",
            ToIntStr(rsize.get(default=0))))
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
            ToIntStr(rsize.get(default=0))))
        field = ctx.add_arg("field")
        ctx.set_return_value(rframe.hget(field))
        return self._rt.register_script(ctx)

    def _pop_frame_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        rframe = RedisHash(
            Strs(ctx.add_key("frame"), ":", ToIntStr(rsize.get(default=0))))
        lcl = ctx.add_local(rframe.hgetall())
        ctx.add(rframe.delete())

        b_then, b_else = ctx.if_(ToNum(rsize.get(default=0)).gt_(0))
        b_then.add(rsize.incrby(-1))
        b_else.add(rsize.delete())

        ctx.set_return_value(lcl)
        return self._rt.register_script(ctx)

    def _get_cascading_script(self) -> ExecFunction:
        ctx = FnContext()
        rsize = RedisVar(ctx.add_key("size"))
        base = ctx.add_local(ctx.add_key("frame"))
        field = ctx.add_arg("field")
        pos = ctx.add_local(ToNum(rsize.get(default=0)))
        res = ctx.add_local(None)
        cur = ctx.add_local(None)
        rframe = RedisHash(cur)

        loop = ctx.while_(res.eq_(None).and_(pos.ge_(0)))
        loop.add(cur.assign(Strs(base, ":", ToIntStr(pos))))
        loop.add(res.assign(rframe.hget(field)))
        loop.add(pos.assign(pos - 1))

        ctx.set_return_value(res)
        return self._rt.register_script(ctx)
