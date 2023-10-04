import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import set_debug_output_length, TaskId
from scattermind.system.client.client import TASK_MAX_RETRIES
from scattermind.system.config.loader import load_test
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
@pytest.mark.parametrize("parallelism", [0, 1, 2, 3])
def test_assertion_error(batch_size: int, parallelism: int) -> None:
    set_debug_output_length(7)
    config = load_test(batch_size=batch_size, parallelism=parallelism)
    config.load_graph({
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
    tasks: list[tuple[TaskId, bool]] = [
        (
            config.enqueue(TaskValueContainer({
                "value": create_tensor(np.array([tix % 3 > 0]), "bool"),
            })),
            tix % 3 > 0,
        )
        for tix in range(20)
    ]
    time_start = time.monotonic()
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    try:
        config.run()
        for task_id, response, expected_result in wait_for_tasks(
                config, tasks):
            real_duration = time.monotonic() - time_start
            status = response["status"]
            task_duration = response["duration"]
            result = response["result"]
            retries = response["retries"]
            error = response["error"]
            assert task_duration <= real_duration
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
                assert ectx["graph_name"].get() == "main"
                assert error["message"].find("value was not true") >= 0
                with pytest.raises(ValueError, match=r"value was not true"):
                    response_ok(response, no_warn=True)
                assert retries == TASK_MAX_RETRIES
            config.clear_task(task_id)
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        print("TEST TEARDOWN!")
        emng = config.get_executor_manager()
        emng.release_all(timeout=1.0)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
