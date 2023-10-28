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


ERROR_CODES: set[ErrorCode] = set(get_args(ErrorCode))


def to_error_code(text: str) -> ErrorCode:
    if text not in ERROR_CODES:
        raise ValueError(f"invalid error code {text}")
    return cast(ErrorCode, text)


ErrorInfo = TypedDict('ErrorInfo', {
    "ctx": ContextInfo,
    "code": ErrorCode,
    "message": str,
    "traceback": list[str],
})


ErrorJSON = TypedDict('ErrorJSON', {
    "ctx": ContextJSON,
    "code": str,
    "message": str,
    "traceback": list[str],
})


def to_error_json(info: ErrorInfo) -> ErrorJSON:
    return {
        "ctx": to_ctx_json(info["ctx"]),
        "code": info["code"],
        "message": info["message"],
        "traceback": info["traceback"],
    }


def from_error_json(err: Mapping) -> ErrorInfo:
    return {
        "ctx": from_ctx_json(err["ctx"]),
        "code": to_error_code(err["code"]),
        "message": err["message"],
        "traceback": err["traceback"],
    }


def build_error_str(error: ErrorInfo, retries: int) -> str:
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
    raise ValueError(build_error_str(error, retries))


def warn_error(error: ErrorInfo, retries: int) -> None:
    print(build_error_str(error, retries), file=sys.stderr)


def to_retry_event(
        error: ErrorInfo,
        retries: int) -> tuple[ContextInfo, 'RetryEvent']:
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
