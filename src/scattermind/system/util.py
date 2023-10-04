from collections.abc import Sequence
from datetime import datetime, timezone


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
    return (from_time - to_time).total_seconds()


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
