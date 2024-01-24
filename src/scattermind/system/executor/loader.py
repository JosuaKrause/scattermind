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
