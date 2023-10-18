from typing import Literal, overload

import torch

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


DataMode = Literal["size"]  # TODO: implement time mode


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


def robj_to_redis(obj: dict) -> str:
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
