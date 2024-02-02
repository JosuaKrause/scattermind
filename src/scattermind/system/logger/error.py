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
"""Functionality for error handling."""
import sys
from collections.abc import Mapping
from typing import cast, get_args, Literal, NoReturn, TYPE_CHECKING, TypedDict

from scattermind.system.logger.context import (
    ContextInfo,
    ContextJSON,
    ctx_format,
    from_ctx_json,
    to_ctx_json,
    to_replace_context,
)


if TYPE_CHECKING:
    from scattermind.system.logger.event import RetryEvent


ErrorCode = Literal[
    "unknown",
    "general_exception",
    "defunc_executor",
    "memory_purge",
    "out_of_memory",
    "uncaught_executor",
]
"""The type of error."""


ERROR_CODES: set[ErrorCode] = set(get_args(ErrorCode))
"""All types of errors."""


def to_error_code(text: str) -> ErrorCode:
    """
    Convert a string to an error code.

    Args:
        text (str): The error code.

    Raises:
        ValueError: If the provided string is not an error code.

    Returns:
        ErrorCode: The error code.
    """
    if text not in ERROR_CODES:
        raise ValueError(f"invalid error code {text}")
    return cast(ErrorCode, text)


ErrorInfo = TypedDict('ErrorInfo', {
    "ctx": ContextInfo,
    "code": ErrorCode,
    "message": str,
    "traceback": list[str],
})
"""Additional context for an error."""


ErrorJSON = TypedDict('ErrorJSON', {
    "ctx": ContextJSON,
    "code": str,
    "message": str,
    "traceback": list[str],
})
"""Additional context for an error as JSON."""


def to_error_json(info: ErrorInfo) -> ErrorJSON:
    """
    Convert an error to a JSONable object.

    Args:
        info (ErrorInfo): The error.

    Returns:
        ErrorJSON: The error in JSONable format.
    """
    return {
        "ctx": to_ctx_json(info["ctx"]),
        "code": info["code"],
        "message": info["message"],
        "traceback": info["traceback"],
    }


def from_error_json(err: Mapping) -> ErrorInfo:
    """
    Parse a JSON object as error.

    Args:
        err (Mapping): The JSON object.

    Returns:
        ErrorInfo: The error object.
    """
    return {
        "ctx": from_ctx_json(err["ctx"]),
        "code": to_error_code(err["code"]),
        "message": err["message"],
        "traceback": err["traceback"],
    }


def build_error_str(error: ErrorInfo, retries: int) -> str:
    """
    Convert an error to a string.

    Args:
        error (ErrorInfo): The error.
        retries (int): The number of retries.

    Returns:
        str: The string.
    """
    message = error["message"]
    code = error["code"]
    ctx_str = ctx_format(error["ctx"])
    tback = error["traceback"]
    tback_str = "\n".join(tback) if tback else ""
    tback_full = f"\nOriginal stacktrace:\n{tback_str}" if tback_str else ""
    return (
        f"Error ({code}) for {ctx_str} with {retries}:\n{message}{tback_full}"
    )


def raise_error(error: ErrorInfo, retries: int) -> NoReturn:
    """
    Raises the error as exception.

    Args:
        error (ErrorInfo): The error.
        retries (int): The number of retries.

    Raises:
        ValueError: The raised exception from the error.
    """
    raise ValueError(build_error_str(error, retries))


def warn_error(error: ErrorInfo, retries: int) -> None:
    """
    Print the error as warning to stderr.

    Args:
        error (ErrorInfo): The error.
        retries (int): The number of retries.
    """
    print(build_error_str(error, retries), file=sys.stderr)


def to_retry_event(
        error: ErrorInfo,
        retries: int) -> tuple[ContextInfo, 'RetryEvent']:
    """
    Convert an error to a retry event.

    Args:
        error (ErrorInfo): The error.
        retries (int): The number of retries.

    Returns:
        tuple[ContextInfo, RetryEvent]: The retry event.
    """
    return (
        to_replace_context(error["ctx"]),
        {
            "name": "retry",
            "code": error["code"],
            "message": error["message"],
            "traceback": error["traceback"],
            "retries": retries,
        },
    )
