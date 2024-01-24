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
"""This module provides some utility functions."""
import base64
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, NoReturn


def is_partial_match(target: str, pattern: str) -> bool:
    """
    Checks whether pattern is a partial match of target. Target is a string
    denoting a hierarchy path separated by '.'. If pattern starts with a '.'
    the check is for any full path segment. Otherwise the pattern is checked
    from the beginning of target and only matches full path segments.

    Examples:
    | Target        | Pattern | Match   |
    | ------------- | ------- | ------- |
    | `foo.bar`     | `foo`   | `True`  |
    | `foobar`      | `foo`   | `False` |
    | `foo.bar`     | `bar`   | `False` |
    | `foo.bar`     | `.bar`  | `True`  |
    | `foo.bar.baz` | `.bar`  | `True`  |
    | `foo.barbaz`  | `.bar`  | `False` |

    Args:
        target (str): The target.
        pattern (str): The pattern.

    Returns:
        bool: Whether the target matches the pattern.
    """
    if pattern.startswith("."):
        if target.endswith(pattern):
            return True
        return target.find(f"{pattern}.") >= 0
    if target == pattern:
        return True
    return target.startswith(f"{pattern}.")


def full_name(cls: type) -> str:
    """
    Return the fully qualified name of the given type.
    Examples: `str`, `scattermind.system.base.Module`

    Args:
        cls (type): The type.

    Returns:
        str: The fully qualified name of the type.
    """
    module = cls.__module__
    qualname = cls.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def shorthand_if_mod(cls: type, mod_prefix: str) -> str:
    """
    Returns the fully qualified name of the given type but returns only the
    last segment if the type is located at mod_prefix.

    Args:
        cls (type): The type.
        mod_prefix (str): The package for shorthands.

    Returns:
        str: The qualified name.
    """
    res = full_name(cls)
    if is_partial_match(res, mod_prefix):
        return f"{cls.__qualname__}"
    return res


def as_int_list(arr: list[int]) -> list[int]:
    """
    Converts all elements of the given list to integers. This is useful for
    processing outside data where we cannot trust that the items in the list
    are actually integers or have the integers represented as strings.

    Args:
        arr (list[int]): The list to convert.

    Returns:
        list[int]: The converted list.
    """
    return [int(elem) for elem in arr]


def as_shape(arr: Sequence[int | None]) -> list[int | None]:
    """
    Converts a sequence into a variable shape. The shape is a list of integers
    where the first element indicates the size of the first dimension etc. The
    shape is variable meaning it can have None values leaving the general size
    of that dimension unspecified until it is used on a concrete instance.

    Args:
        arr (Sequence[int  |  None]): The sequence to convert.

    Returns:
        list[int | None]: The variable shape.
    """
    return [None if elem is None else int(elem) for elem in arr]


def now() -> datetime:
    """
    Computes the current time with UTC timezone.

    Returns:
        datetime: A timezone aware instance of now.
    """
    return datetime.now(timezone.utc).astimezone()


def fmt_time(when: datetime) -> str:
    """
    Formats a timestamp as ISO formatted string.

    Args:
        when (datetime): The timestamp.

    Returns:
        str: The formatted string.
    """
    return when.isoformat()


def get_time_str() -> str:
    """
    Get the current time as ISO formatted string.

    Returns:
        str: The current time in ISO format.
    """
    return fmt_time(now())


def parse_time_str(time_str: str) -> datetime:
    """
    Parses an ISO formatted string representing a timestamp.

    Args:
        time_str (str): The string.

    Returns:
        datetime: The timestamp.
    """
    return datetime.fromisoformat(time_str)


def time_diff(from_time: datetime, to_time: datetime) -> float:
    """
    Computes the number of seconds between two timestamps.

    Args:
        from_time (datetime): The first timestamp.
        to_time (datetime): The second timestamp.

    Returns:
        float: The number of seconds between the timestamps. Can be negative if
            from_time is after to_time.
    """
    return (to_time - from_time).total_seconds()


def seconds_since(time_str: str) -> float:
    """
    Computes the number of seconds since the given ISO formatted time string.

    Args:
        time_str (str): The ISO formatted time string.

    Returns:
        float: The number of seconds since the time string. Can be negative
            if the specified time is in the future.
    """
    return time_diff(parse_time_str(time_str), now())


def to_bool(text: str | None) -> bool:
    """
    Makes a best effort conversion of the value to a boolean. If the value is
    None it is interpreted as False. If the value is a number or can be parsed
    as number it is interpreted as False exactly if the number is 0. Otherwise,
    any string except for case insensitive `true` values is interpreted as
    False.

    Args:
        text (str | None): The value to convert.

    Returns:
        bool: The converted boolean.
    """
    if text is None:
        return False
    try:
        return int(text) > 0
    except ValueError:
        pass
    return f"{text}".lower() == "true"


def as_base85(value: bytes) -> str:
    """
    Converts a byte sequence into a base 85 encoded string.

    Args:
        value (bytes): The byte sequence.

    Returns:
        str: The base 85 encoded string.
    """
    return base64.b85encode(value).decode("ascii")


def from_base85(text: str) -> bytes:
    """
    Converts a base 85 encoded string into a byte sequence.

    Args:
        text (str): The base 85 encoded string.

    Returns:
        bytes: The byte sequence.
    """
    return base64.b85decode(text)


def get_bytes_hash(value: bytes) -> str:
    """
    Computes a hash for the given byte sequence. The length of the resulting
    hash string can be retrieved via :py:function::`bytes_hash_size`.

    Args:
        value (bytes): The byte sequence.

    Returns:
        str: The hash.
    """
    blake = hashlib.blake2b(digest_size=32)
    blake.update(value)
    return blake.hexdigest()


def bytes_hash_size() -> int:
    """
    The size of the hash string generated by :py:function::`get_bytes_hash`.

    Returns:
        int: The length of the hash string.
    """
    return 64


def get_text_hash(text: str) -> str:
    """
    Computes a hash for the given text. The length of the resulting
    hash string can be retrieved via :py:function::`text_hash_size`.

    Args:
        text (str): The text.

    Returns:
        str: The hash.
    """
    blake = hashlib.blake2b(digest_size=32)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def text_hash_size() -> int:
    """
    The size of the hash string generated by :py:function::`get_text_hash`.

    Returns:
        int: The length of the hash string.
    """
    return 64


def get_short_hash(text: str) -> str:
    """
    Computes a short hash for the given text. The length of the resulting
    hash string can be retrieved via :py:function::`short_hash_size`.

    Args:
        text (str): The text.

    Returns:
        str: The hash.
    """
    blake = hashlib.blake2b(digest_size=4)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def short_hash_size() -> int:
    """
    The size of the short hash string generated by
    :py:function::`short_text_hash`.

    Returns:
        int: The length of the hash string.
    """
    return 8


BUFF_SIZE = 65536  # 64KiB
"""The buffer size for computing hashes for file contents."""


def get_file_hash(fname: str) -> str:
    """
    Computes a hash for the content of the given filename. The length of the
    resulting hash string can be retrieved via :py:function::`file_hash_size`.

    Args:
        fname (str): The filename.

    Returns:
        str: The hash.
    """
    blake = hashlib.blake2b(digest_size=32)
    with open(fname, "rb") as fin:
        while True:
            buff = fin.read(BUFF_SIZE)
            if not buff:
                break
            blake.update(buff)
    return blake.hexdigest()


def file_hash_size() -> int:
    """
    The size of the hash string generated by :py:function::`get_file_hash`.

    Returns:
        int: The length of the hash string.
    """
    return 64


def report_json_error(err: json.JSONDecodeError) -> NoReturn:
    """
    Reports a JSON error by adding additional information about where the
    error is located in the JSON.

    Args:
        err (json.JSONDecodeError): The original error.

    Raises:
        ValueError: The amended error.
    """
    raise ValueError(
        f"JSON parse error ({err.lineno}:{err.colno}): "
        f"{repr(err.doc)}") from err


def json_compact(obj: Any) -> str:
    """
    Creates a compact JSON from the given object.

    Args:
        obj (Any): The object.

    Returns:
        str: A JSON without any spaces or new lines.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        indent=None,
        separators=(",", ":"))


def json_read(data: str) -> Any:
    """
    Parses data as JSON.

    Args:
        data (str): The data to parse.

    Raises:
        ValueError: If the data couldn't be parsed.

    Returns:
        Any: The JSON object. Make sure to validate the expected layout.
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        report_json_error(e)
