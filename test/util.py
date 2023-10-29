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


def is_github_action() -> bool:
    global IS_GH_ACTION

    if IS_GH_ACTION is None:
        IS_GH_ACTION = to_bool(os.getenv("GITHUB_ACTIONS"))
    return IS_GH_ACTION


def skip_on_gha_if(condition: bool, reason: str) -> None:
    if is_github_action() and condition:
        pytest.skip(reason)


def wait_for_tasks(
        config: 'Config',
        tasks: list[tuple['TaskId', T]],
        *,
        timeinc: float = 0.1,
        timeout: float = 0.5,
        ) -> Iterable[tuple['TaskId', 'ResponseObject', T]]:
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
