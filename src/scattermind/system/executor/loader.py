# Copyright (C) 2024 Josua Krause
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loads an executor manager."""
from collections.abc import Callable
from typing import Literal, TypedDict

from redipy import RedisConfig

from scattermind.system.base import ExecutorId
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.plugins import load_plugin


SingleExecutorManagerModule = TypedDict('SingleExecutorManagerModule', {
    "name": Literal["single"],
    "batch_size": int,
})
"""A singular executor that terminates once all tasks are processed."""
ThreadExecutorManagerModule = TypedDict('ThreadExecutorManagerModule', {
    "name": Literal["thread"],
    "batch_size": int,
    "parallelism": int,
    "sleep_on_idle": float,
    "reclaim_sleep": float,
})
"""A thread executor that continues executing until the process is terminated
or all executors are released. `parallelism` defines the number of worker
threads."""
RedisExecutorManagerModule = TypedDict('RedisExecutorManagerModule', {
    "name": Literal["redis"],
    "batch_size": int,
    "sleep_on_idle": float,
    "reclaim_sleep": float,
    "heartbeat_time": float,
    "cfg": RedisConfig,
})
"""A redis based executor. Ideal for distributed workers."""


ExecutorManagerModule = (
    SingleExecutorManagerModule
    | ThreadExecutorManagerModule
    | RedisExecutorManagerModule
)
"""Executor manager configuration."""


def _load_executor_manager(
        exec_gen: Callable[[], ExecutorId],
        module: ExecutorManagerModule,
        ) -> tuple[Callable[[], ExecutorManager], bool]:
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(ExecutorManager, f"{kwargs.pop('name')}")
        batch_size = int(f"{kwargs.pop('batch_size')}")
        return (
            lambda: plugin(exec_gen(), batch_size=batch_size, **kwargs),
            plugin.allow_parallel(),
        )
    if module["name"] == "single":
        from scattermind.system.executor.single import SingleExecutorManager
        return (
            lambda: SingleExecutorManager(
                exec_gen(), batch_size=module["batch_size"]),
            SingleExecutorManager.allow_parallel(),
        )
    if module["name"] == "thread":
        from scattermind.system.executor.thread import ThreadExecutorManager
        return (
            lambda: ThreadExecutorManager(
                exec_gen(),
                batch_size=module["batch_size"],
                sleep_on_idle=module["sleep_on_idle"],
                reclaim_sleep=module["reclaim_sleep"]),
            ThreadExecutorManager.allow_parallel(),
        )
    if module["name"] == "redis":
        from scattermind.system.executor.redis import RedisExecutorManager
        return (
            lambda: RedisExecutorManager(
                exec_gen(),
                batch_size=module["batch_size"],
                sleep_on_idle=module["sleep_on_idle"],
                reclaim_sleep=module["reclaim_sleep"],
                heartbeat_time=module["heartbeat_time"],
                cfg=module["cfg"]),
            RedisExecutorManager.allow_parallel(),
        )
    raise ValueError(f"unknown executor manager: {module['name']}")


def load_executor_manager(
        exec_gen: Callable[[], ExecutorId],
        module: ExecutorManagerModule) -> ExecutorManager:
    """
    Load an executor manager.

    Args:
        exec_gen (Callable[[], ExecutorId]): Generator for new executor ids.
            This function might be called multiple times depending on the
            executor manager.
        module (ExecutorManagerModule): The executor manager configuration.
            The `name` field can be a fully qualified python module to load
            a plugin.

    Raises:
        ValueError: If configuration is invalid.

    Returns:
        ExecutorManager: The executor manager.
    """
    constructor, has_parallelism = _load_executor_manager(exec_gen, module)
    local_parallelism = int(module.get("parallelism", 1))  # type: ignore
    if has_parallelism and local_parallelism > 1:
        ems = [constructor() for _ in range(local_parallelism)]
        return ems[0]
    return constructor()
