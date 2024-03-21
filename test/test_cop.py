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
"""Test basic execution graphs."""
import numpy as np
import pytest

from scattermind.system.base import TaskId
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace
from scattermind.system.response import (
    response_ok,
    TASK_STATUS_DONE,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("is_redis", [False, True])
def test_cop(base: list[list[float]], batch_size: int, is_redis: bool) -> None:
    """
    Test constant operator.

    Args:
        base (list[list[float]]): The base value.
        batch_size (int): The batch size for processing.
        is_redis (bool): Whether to use redis.
    """
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
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue_task(
                "cop",
                {
                    "value": (np.array(base) * tix).astype(np.float32),
                }),
            np.array(base) * tix * 2.0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    config.run(force_no_block=False)  # NOTE: we only use single here
    for task_id, expected_result in tasks:
        response = config.get_response(task_id)
        response_ok(response, no_warn=True)
        assert response["status"] == TASK_STATUS_READY
        assert response["ns"] == ns
        result = response["result"]
        assert result is not None
        assert list(result["value"].shape) == shape
        np.testing.assert_allclose(as_numpy(result["value"]), expected_result)
        assert config.get_status(task_id) == TASK_STATUS_DONE
        config.clear_task(task_id)
        assert config.get_namespace(task_id) is None
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("is_redis", [False, True])
def test_cop_chain(
        base: list[list[float]], batch_size: int, is_redis: bool) -> None:
    """
    Test chaining constant operators.

    Args:
        base (list[list[float]]): The base value.
        batch_size (int): The batch size for processing.
        is_redis (bool): Whether to use redis.
    """
    shape = [len(base), len(base[0])]
    config = load_test(batch_size=batch_size, is_redis=is_redis)
    ns = config.load_graph({
        "graphs": [
            {
                "name": "copchain",
                "description": f"batch_size={batch_size},base={base}",
                "input": "node_0",
                "input_format": {
                    "value": ("float", shape),
                },
                "output_format": {
                    "value": ("float", shape),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "constant_op",
                        "args": {
                            "op": "mul",
                            "const": 2,
                            "input": ("float", shape),
                        },
                        "outs": {
                            "out": "node_1",
                        },
                        "vmap": {
                            "value": ":value",
                        },
                    },
                    {
                        "name": "node_1",
                        "kind": "constant_op",
                        "args": {
                            "op": "add",
                            "const": 1,
                            "input": ("float", shape),
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "value": "node_0:value",
                        },
                    },
                ],
                "vmap": {
                    "value": "node_1:value",
                },
            },
        ],
        "entry": "copchain",
    })
    assert ns == GNamespace("copchain")
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue_task(
                "copchain",
                {
                    "value": (np.array(base) * tix).astype(np.float32),
                }),
            np.array(base) * tix * 2.0 + 1.0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    config.run(force_no_block=False)  # NOTE: we only use single here
    for task_id, expected_result in tasks:
        response = config.get_response(task_id)
        response_ok(response, no_warn=True)
        assert response["status"] == TASK_STATUS_READY
        assert response["ns"] == ns
        result = response["result"]
        assert result is not None
        assert list(result["value"].shape) == shape
        np.testing.assert_allclose(as_numpy(result["value"]), expected_result)
        assert config.get_status(task_id) == TASK_STATUS_DONE
        config.clear_task(task_id)
        assert config.get_namespace(task_id) is None
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None
