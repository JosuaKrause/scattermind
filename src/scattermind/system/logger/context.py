import contextlib
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import TypedDict, TypeVar

from typing_extensions import NotRequired

from scattermind.system.base import ExecutorId, GraphId, NodeId, TaskId
from scattermind.system.names import GName, NName


T = TypeVar('T')


TH_LOCAL = threading.local()


ContextInfo = TypedDict('ContextInfo', {
    "executor": NotRequired[ExecutorId | None],
    "task": NotRequired[TaskId | None],
    "graph": NotRequired[GraphId | None],
    "graph_name": NotRequired[GName | None],
    "node": NotRequired[NodeId | None],
    "node_name": NotRequired[NName | None],
})


ContextJSON = TypedDict('ContextJSON', {
    "executor": NotRequired[str | None],
    "task": NotRequired[str | None],
    "graph": NotRequired[str | None],
    "graph_name": NotRequired[str | None],
    "node": NotRequired[str | None],
    "node_name": NotRequired[str | None],
})


def to_ctx_json(info: ContextInfo) -> ContextJSON:

    def mapval(val: T | None, transform: Callable[[T], str]) -> str | None:
        if val is None:
            return None
        return transform(val)

    return {
        "executor": mapval(info.get("executor"), lambda e: e.to_parseable()),
        "task": mapval(info.get("task"), lambda t: t.to_parseable()),
        "graph": mapval(info.get("graph"), lambda g: g.to_parseable()),
        "graph_name": mapval(info.get("graph_name"), lambda n: n.get()),
        "node": mapval(info.get("node"), lambda n: n.to_parseable()),
        "node_name": mapval(info.get("node_name"), lambda n: n.get()),
    }


def from_ctx_json(ctx: Mapping) -> ContextInfo:

    def mapval(val: str | None, transform: Callable[[str], T]) -> T | None:
        if val is None:
            return None
        return transform(val)

    return {
        "executor": mapval(ctx.get("executor"), ExecutorId.parse),
        "task": mapval(ctx.get("task"), TaskId.parse),
        "graph": mapval(ctx.get("graph"), GraphId.parse),
        "graph_name": mapval(ctx.get("graph_name"), GName),
        "node": mapval(ctx.get("node"), NodeId.parse),
        "node_name": mapval(ctx.get("node_name"), NName),
    }


NAME_CTX = "ctx"
NAME_PREEXC_CTX = "preexc_ctx"


def _get_context() -> ContextInfo:
    res: ContextInfo | None = getattr(TH_LOCAL, NAME_CTX, None)
    if res is None:
        res = {}
        setattr(TH_LOCAL, NAME_CTX, res)
    return res


def _set_context(ctx: ContextInfo) -> None:
    setattr(TH_LOCAL, NAME_CTX, ctx)


def _get_preexc_context() -> ContextInfo:
    res: ContextInfo | None = getattr(TH_LOCAL, NAME_PREEXC_CTX, None)
    if res is None:
        res = {}
        setattr(TH_LOCAL, NAME_PREEXC_CTX, res)
    return res


def _set_preexc_context(ctx: ContextInfo) -> None:
    setattr(TH_LOCAL, NAME_PREEXC_CTX, ctx)


@contextlib.contextmanager
def add_context(add_info: ContextInfo) -> Iterator[None]:
    old_ctx = _get_context()
    new_ctx = old_ctx.copy()
    new_ctx.update(add_info)
    success = False
    try:
        _set_context(new_ctx)
        _set_preexc_context(new_ctx)
        yield
        success = True
    finally:
        _set_context(old_ctx)
        if success:
            _set_preexc_context(old_ctx)


def get_ctx() -> ContextInfo:
    return _get_context().copy()


def ctx_fmt() -> str:
    return ctx_format(get_ctx())


def get_preexc_ctx() -> ContextInfo:
    return _get_preexc_context().copy()


def preexc_ctx_fmt() -> str:
    return ctx_format(get_preexc_ctx())


def ctx_format(ctx: ContextInfo) -> str:
    executor = ctx.get("executor")
    task = ctx.get("task")
    graph_id = ctx.get("graph")
    graph_name = ctx.get("graph_name")
    node_id = ctx.get("node")
    node_name = ctx.get("node_name")
    if node_id or node_name or graph_id or graph_name:
        node_id_str = "" if node_id is None else f"({node_id})"
        gid_str = "" if graph_id is None else f"({graph_id})"
        node_name_str = "" if node_name is None else f"{node_name.get()}"
        gname_str = "" if graph_name is None else f"{graph_name.get()}"
        graph_str = f" {gname_str}{gid_str}" if gname_str or gid_str else ""
        location = f" in {node_name_str}{node_id_str}{graph_str}"
    else:
        location = ""
    executor_str = (
        f"{ExecutorId.prefix()}[unknown]"
        if executor is None
        else f"{executor}"
    )
    task_str = (
        f"{TaskId.prefix()}[unknown]"
        if task is None
        else f"{task}"
    )
    return f"{executor_str} {task_str}{location}"


def to_replace_context(ctx: ContextInfo) -> ContextInfo:
    return {
        "executor": ctx.get("executor"),
        "task": ctx.get("task"),
        "graph": ctx.get("graph"),
        "graph_name": ctx.get("graph_name"),
        "node": ctx.get("node"),
        "node_name": ctx.get("node_name"),
    }
