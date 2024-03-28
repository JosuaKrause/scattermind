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
"""Test errors in nodes."""
import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import set_debug_output_length, TaskId
from scattermind.system.client.client import TASK_MAX_RETRIES
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    response_ok,
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy, create_tensor


@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("parallelism", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("is_redis", [False, True])
def test_assertion_error(
        batch_size: int, parallelism: int, is_redis: bool) -> None:
    """
    Test causing an assertion error for some tasks.

    Args:
        batch_size (int): The batch size.
        parallelism (int): The parallelism.
        is_redis (bool): Whether to use redis.
    """
    set_debug_output_length(7)
    config = load_test(
        batch_size=batch_size, parallelism=parallelism, is_redis=is_redis)
    ns = config.load_graph({
        "graphs": [
            {
                "name": "main",
                "description":
                    f"batch_size={batch_size};parallelism={parallelism}",
                "input": "node_0",
                "input_format": {
                    "value": ("bool", [1]),
                },
                "output_format": {
                    "value": ("bool", [1]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "if_op",
                        "outs": {
                            "then": None,
                            "else": "node_1",
                        },
                        "vmap": {
                            "condition": ":value",
                        },
                    },
                    {
                        "name": "node_1",
                        "kind": "assertion_error",
                        "args": {
                            "msg": "value was not true",
                        },
                    },
                ],
                "vmap": {
                    "value": ":value",
                },
            },
        ],
        "entry": "main",
    })
    assert ns == GNamespace("main")
    time_start = time.monotonic()
    tasks: list[tuple[TaskId, bool]] = [
        (
            config.enqueue(
                ns,
                TaskValueContainer({
                    "value": create_tensor(
                        np.array([tix % 3 > 0]), dtype="bool"),
                })),
            tix % 3 > 0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    try:
        config.run(force_no_block=True)
        for task_id, response, expected_result in wait_for_tasks(
                config, tasks):
            real_duration = time.monotonic() - time_start
            status = response["status"]
            task_ns = response["ns"]
            task_duration = response["duration"]
            result = response["result"]
            retries = response["retries"]
            error = response["error"]
            assert task_duration <= real_duration
            assert task_ns == ns
            if expected_result:
                response_ok(response, no_warn=True)
                assert status == TASK_STATUS_READY
                assert result is not None
                assert list(result["value"].shape) == [1]
                assert retries == 0
                assert error is None
                np.testing.assert_allclose(
                    as_numpy(result["value"]), np.array([expected_result]))
                assert config.get_status(task_id) == TASK_STATUS_DONE
            else:
                assert status == TASK_STATUS_ERROR
                assert result is None
                assert error is not None
                assert error["code"] == "general_exception"
                ectx = error["ctx"]
                assert ectx["node_name"] is not None
                assert ectx["node_name"].get() == "node_1"
                assert ectx["graph_name"] is not None
                assert ectx["graph_name"].get_namespace().get() == "main"
                assert ectx["graph_name"].get_name().get() == "main"
                assert error["message"].find("value was not true") >= 0
                tback = error["traceback"]
                assert "Traceback" in tback[0]
                assert "line" in tback[-2]
                assert "ValueError: value was not true" in tback[-1]
                with pytest.raises(ValueError, match=r"value was not true"):
                    response_ok(response, no_warn=True)
                assert retries == TASK_MAX_RETRIES
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        print("TEST TEARDOWN!")
        emng = config.get_executor_manager()
        emng.release_all(timeout=0.1)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 3, 5, 12, 30])
@pytest.mark.parametrize("is_redis", [False, True])
def test_ghost(
        base: list[list[float]],
        batch_size: int,
        is_redis: bool) -> None:
    """
    Test handling of ghost tasks.

    Args:
        base (list[list[float]]): The base value.
        batch_size (int): The batch size for processing.
        is_redis (bool): Whether to use redis.
    """
    set_debug_output_length(7)
    shape = [len(base), len(base[0])]
    config = load_test(batch_size=batch_size, is_redis=is_redis)
    ns = config.load_graph({
        "graphs": [
            {
                "name": "cop",
                "description": f"batch_size={batch_size},base={base}",
                "input": "node",
                "input_format": {
                    "value": ("float", shape),
                },
                "output_format": {
                    "value": ("float", shape),
                },
                "nodes": [
                    {
                        "name": "node",
                        "kind": "constant_op",
                        "args": {
                            "op": "mul",
                            "const": 2,
                            "input": ("float", shape),
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "value": ":value",
                        },
                    },
                ],
                "vmap": {
                    "value": "node:value",
                },
            },
        ],
        "entry": "cop",
    })
    assert ns == GNamespace("cop")
    time_start = time.monotonic()
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue_task(
                "cop",
                {
                    "value": (np.array(base) * tix).astype(np.float32),
                }),
            np.array(base) * tix * 2.0,
        )
        for tix in range(12)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    bad_tasks: set[TaskId] = set()
    for task_id, _ in tasks[::2]:
        config.clear_task(task_id)  # NOTE: prematurely clean tasks
        assert config.get_namespace(task_id) is None
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None
        bad_tasks.add(task_id)
    try:
        config.run(force_no_block=False)  # NOTE: we only use single here
        for task_id, response, expected_result in wait_for_tasks(
                config, tasks):
            if task_id in bad_tasks:
                assert response["status"] == TASK_STATUS_UNKNOWN
                assert config.get_namespace(task_id) is None
                assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
                assert config.get_result(task_id) is None
                continue
            real_duration = time.monotonic() - time_start
            status = response["status"]
            task_ns = response["ns"]
            task_duration = response["duration"]
            result = response["result"]
            assert status == TASK_STATUS_READY
            assert result is not None
            assert list(result["value"].shape) == shape
            np.testing.assert_allclose(as_numpy(
                result["value"]), expected_result)
            assert task_duration <= real_duration
            assert task_ns == ns
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        print("TEST TEARDOWN!")
        emng = config.get_executor_manager()
        emng.release_all(timeout=0.1)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
