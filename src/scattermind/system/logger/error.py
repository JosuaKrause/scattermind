import sys
from typing import Literal, NoReturn, TYPE_CHECKING, TypedDict

from scattermind.system.logger.context import (
    ContextInfo,
    ctx_format,
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


ErrorInfo = TypedDict('ErrorInfo', {
    "ctx": ContextInfo,
    "code": ErrorCode,
    "message": str,
    "traceback": list[str],
})


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
