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
"""Loads an executor manager."""
from collections.abc import Callable
from typing import Literal, TypedDict

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


ExecutorManagerModule = (
    SingleExecutorManagerModule
    | ThreadExecutorManagerModule
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
