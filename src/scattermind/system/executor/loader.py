from collections.abc import Callable
from typing import Literal, TypedDict

from scattermind.system.base import ExecutorId
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.plugins import load_plugin


SingleExecutorManagerModule = TypedDict('SingleExecutorManagerModule', {
    "name": Literal["single"],
    "batch_size": int,
})
ThreadExecutorManagerModule = TypedDict('ThreadExecutorManagerModule', {
    "name": Literal["thread"],
    "batch_size": int,
    "parallelism": int,
    "sleep_on_idle": float,
})


ExecutorManagerModule = (
    SingleExecutorManagerModule
    | ThreadExecutorManagerModule
)


def load_executor_manager(
        exec_gen: Callable[[], ExecutorId],
        module: ExecutorManagerModule) -> ExecutorManager:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ExecutorManager, f"{kwargs.pop('name')}")
        batch_size = int(f"{kwargs.pop('batch_size')}")
        return plugin(exec_gen(), batch_size=batch_size, **kwargs)
    if module["name"] == "single":
        from scattermind.system.executor.single import SingleExecutorManager
        return SingleExecutorManager(
            exec_gen(), batch_size=module["batch_size"])
    if module["name"] == "thread":
        from scattermind.system.executor.thread import ThreadExecutorManager
        tems = [
            ThreadExecutorManager(
                exec_gen(),
                batch_size=module["batch_size"],
                sleep_on_idle=module["sleep_on_idle"])
            for _ in range(module["parallelism"])
        ]
        return tems[0]
    raise ValueError(f"unknown executor manager: {module['name']}")
