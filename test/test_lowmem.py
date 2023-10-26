import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import TaskId
from scattermind.system.client.client import TASK_MAX_RETRIES
from scattermind.system.config.loader import load_test
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy, create_tensor


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 10, 20])
def test_low_mem(base: list[list[float]], batch_size: int) -> None:
    # FIXME work out a way to enable redis here
    shape = [len(base), len(base[0])]
    config = load_test(
        batch_size=batch_size, max_store_size=200, is_redis=False)
    config.load_graph({
        "graphs": [
            {
                "name": "lowmem",
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
        "entry": "lowmem",
    })
    time_start = time.monotonic()
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue(TaskValueContainer({
                "value": create_tensor(np.array(base) * tix, "float"),
            })),
            np.array(base) * tix * 2.0 + 1.0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    config.run()
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
        result = response["result"]
        task_duration = response["duration"]
        assert task_duration <= real_duration
        if (status == "error" if is_variable else has_error):
            retries = response["retries"]
            error = response["error"]
            assert error is not None
            assert status == TASK_STATUS_ERROR
            assert result is None
            assert retries == TASK_MAX_RETRIES
            assert error["code"] in ("out_of_memory", "memory_purge")
            config.clear_task(task_id)
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
