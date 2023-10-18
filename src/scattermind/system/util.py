import base64
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any


def is_partial_match(target: str, pattern: str) -> bool:
    if pattern.startswith("."):
        if target.endswith(pattern):
            return True
        return target.find(f"{pattern}.") >= 0
    if target == pattern:
        return True
    return target.startswith(f"{pattern}.")


def full_name(cls: type) -> str:
    module = cls.__module__
    qualname = cls.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def shorthand_if_mod(cls: type, mod_prefix: str) -> str:
    res = full_name(cls)
    if is_partial_match(res, mod_prefix):
        return f"{cls.__qualname__}"
    return res


def as_int_list(arr: list[int]) -> list[int]:
    return [int(elem) for elem in arr]


def as_shape(arr: Sequence[int | None]) -> list[int | None]:
    return [None if elem is None else int(elem) for elem in arr]


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def fmt_time(when: datetime) -> str:
    return when.isoformat()


def get_time_str() -> str:
    return fmt_time(now())


def parse_time_str(time_str: str) -> datetime:
    return datetime.fromisoformat(time_str)


def time_diff(from_time: datetime, to_time: datetime) -> float:
    return (to_time - from_time).total_seconds()


def seconds_since(time_str: str) -> float:
    return time_diff(parse_time_str(time_str), now())


def to_bool(text: str | None) -> bool:
    if text is None:
        return False
    try:
        return int(text) > 0
    except ValueError:
        pass
    return f"{text}".lower() == "true"


def as_base85(value: bytes) -> str:
    return base64.b85encode(value).decode("ascii")


def from_base85(text: str) -> bytes:
    return base64.b85decode(text)


def get_bytes_hash(value: bytes) -> str:
    blake = hashlib.blake2b(digest_size=32)
    blake.update(value)
    return blake.hexdigest()


def bytes_hash_size() -> int:
    return 64


def get_text_hash(text: str) -> str:
    blake = hashlib.blake2b(digest_size=32)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def text_hash_size() -> int:
    return 64


def get_short_hash(text: str) -> str:
    blake = hashlib.blake2b(digest_size=4)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def short_hash_size() -> int:
    return 8


BUFF_SIZE = 65536  # 64KiB


def get_file_hash(fname: str) -> str:
    blake = hashlib.blake2b(digest_size=32)
    with open(fname, "rb") as fin:
        while True:
            buff = fin.read(BUFF_SIZE)
            if not buff:
                break
            blake.update(buff)
    return blake.hexdigest()


def file_hash_size() -> int:
    return 64


def report_json_error(err: json.JSONDecodeError) -> None:
    raise ValueError(
        f"JSON parse error ({err.lineno}:{err.colno}): "
        f"{repr(err.doc)}") from err


def json_compact(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        indent=None,
        separators=(",", ":"))


def json_read(data: str) -> Any:
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        report_json_error(e)
        raise e
