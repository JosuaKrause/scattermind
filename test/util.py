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
"""Utility functions for unit tests."""
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import pytest

from scattermind.system.util import to_bool


if TYPE_CHECKING:
    from scattermind.system.base import TaskId
    from scattermind.system.config.config import Config
    from scattermind.system.response import ResponseObject


T = TypeVar('T')


IS_GH_ACTION: bool | None = None
"""Lookup value whether the current process is run by github actions."""


def is_github_action() -> bool:
    """
    Whether the current process is run by github actions.

    Returns:
        bool: True, if the execution happens from a github actions runner.
    """
    global IS_GH_ACTION  # pylint: disable=global-statement

    if IS_GH_ACTION is None:
        IS_GH_ACTION = to_bool(os.getenv("GITHUB_ACTIONS"))
    return IS_GH_ACTION


def skip_on_gha_if(condition: bool, reason: str) -> None:
    """
    Instructs the test to be skipped if it is run on github actions.

    Args:
        condition (bool): If True, the test will be skipped on github actions.
        reason (str): The reason for the skip.
    """
    if is_github_action() and condition:
        pytest.skip(reason)


def wait_for_tasks(
        config: 'Config',
        tasks: list[tuple['TaskId', T]],
        *,
        timeinc: float = 0.1,
        timeout: float = 5.0,
        ) -> Iterable[tuple['TaskId', 'ResponseObject', T]]:
    """
    Wait for scattermind tasks to complete. The function does not check by
    itself whether the expected result was returned. But the function ensures
    that all tasks were returned.

    Args:
        config (Config): The configuration.
        tasks (list[tuple[TaskId, T]]): A list of tasks and expected results.
        timeinc (float, optional): The increment of internal waiting
            between checks. Defaults to 0.1.
        timeout (float, optional): The maximum time to wait for any task
            to complete. Defaults to 5.0.

    Yields:
        tuple[TaskId, ResponseObject, T]: The task, its response, and the
            expected result.
    """
    task_ids: list['TaskId'] = []
    expected: dict['TaskId', T] = {}
    for task_id, expected_result in tasks:
        assert task_id not in expected
        expected[task_id] = expected_result
        task_ids.append(task_id)
    seen: set['TaskId'] = set()
    for task_id, response in config.wait_for(
            task_ids, timeinc=timeinc, timeout=timeout):
        expected_result = expected[task_id]
        yield task_id, response, expected_result
        seen.add(task_id)
    assert seen == set(task_ids)
