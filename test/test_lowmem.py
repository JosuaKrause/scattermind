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
"""Test scattermind under low memory conditions."""
import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import TaskId
from scattermind.system.client.client import TASK_MAX_RETRIES
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 10, 20])
def test_low_mem(base: list[list[float]], batch_size: int) -> None:
    """
    Test scattermind under low memory conditions for the payload data store.

    Args:
        base (list[list[float]]): The base value.
        batch_size (int): The batch size.
    """
    # FIXME work out a way to enable redis here
    shape = [len(base), len(base[0])]
    config = load_test(
        batch_size=batch_size, max_store_size=200, is_redis=False)
    ns = config.load_graph({
        "graphs": [
            {
                "name": "lowmem",
                "description": f"{batch_size=},{base=}",
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
        "entry": "lowmem",
    })
    assert ns == GNamespace("lowmem")
    time_start = time.monotonic()
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue_task(
                ns,
                {
                    "value": (np.array(base) * tix).astype(np.float32),
                }),
            np.array(base) * tix * 2.0 + 1.0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    # NOTE: we only use single here
    config.run(force_no_block=False, no_reclaim=True)
    has_error = batch_size >= 3 if len(base) < 2 else batch_size >= 2
    is_variable = (batch_size, len(base)) in (
        (2, 2),
        (3, 1),
        (3, 2),
        (10, 1),
        (10, 2),
        (20, 1),
        (20, 2),
    )
    seen_error = False
    seen_result = False
    for task_id, response, expected_result in wait_for_tasks(config, tasks):
        real_duration = time.monotonic() - time_start
        print(response)
        status = response["status"]
        task_ns = response["ns"]
        result = response["result"]
        task_duration = response["duration"]
        assert task_duration <= real_duration
        assert task_ns == ns
        if (status == "error" if is_variable else has_error):
            retries = response["retries"]
            error = response["error"]
            assert error is not None
            assert status == TASK_STATUS_ERROR
            assert result is None
            assert retries == TASK_MAX_RETRIES
            assert error["code"] in ("out_of_memory", "memory_purge")
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
            seen_error = True
        else:
            assert status == TASK_STATUS_READY
            assert result is not None
            assert list(result["value"].shape) == shape
            np.testing.assert_allclose(
                as_numpy(result["value"]), expected_result)
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
            seen_result = True
    if is_variable:
        assert seen_error
        assert seen_result
    elif has_error:
        assert seen_error
        assert not seen_result
    else:
        assert not seen_error
        assert seen_result
