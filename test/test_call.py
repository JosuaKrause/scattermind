import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import set_debug_output_length, TaskId
from scattermind.system.config.loader import load_test
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    response_ok,
    TASK_STATUS_DONE,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy, create_tensor


@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("parallelism", [0, 1, 2, 3])
def test_simple_call(batch_size: int, parallelism: int) -> None:
    set_debug_output_length(7)
    config = load_test(batch_size=batch_size, parallelism=parallelism)
    writer = config.get_roa_writer()
    cpath = "procedure_const_0"
    with writer.open_write(cpath) as fout:
        constant_0 = fout.as_data_str(fout.write_tensor(
            create_tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), "float")))
    config.load_graph({
        "graphs": [
            {
                "name": "main",
                "description":
                    f"batch_size={batch_size};parallelism={parallelism}",
                "input": "node_0",
                "input_format": {
                    "value": ("float", [2, 2]),
                },
                "output_format": {
                    "value": ("float", [2, 2]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "constant_op",
                        "args": {
                            "op": "mul",
                            "const": 2,
                            "input": ("float", [2, 2]),
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
                        "kind": "call",
                        "args": {
                            "graph": "procedure",
                            "args": {
                                "mat": ("float", [2, 2]),
                            },
                            "ret": {
                                "mat": ("float", [2, 2]),
                            },
                        },
                        "outs": {
                            "out": "node_2",
                        },
                        "vmap": {
                            "mat": "node_0:value",
                        },
                    },
                    {
                        "name": "node_2",
                        "kind": "constant_op",
                        "args": {
                            "op": "add",
                            "const": 1,
                            "input": ("float", [2, 2]),
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "value": "node_1:mat",
                        },
                    },
                ],
                "vmap": {
                    "value": "node_2:value",
                },
            },
            {
                "name": "procedure",
                "input": "node_0",
                "input_format": {
                    "mat": ("float", [2, 2]),
                },
                "output_format": {
                    "value": ("float", [2, 2]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "load_constant",
                        "args": {
                            "data": constant_0,
                            "ret": ("float", [2, 2]),
                        },
                        "outs": {
                            "out": "node_1",
                        },
                    },
                    {
                        "name": "node_1",
                        "kind": "mat_square",
                        "args": {
                            "size": 2,
                        },
                        "outs": {
                            "out": "node_2",
                        },
                        "vmap": {
                            "value": ":mat",
                        },
                    },
                    {
                        "name": "node_2",
                        "kind": "bin_op",
                        "args": {
                            "op": "mul",
                            "input": ("float", [2, 2]),
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "left": "node_1:value",
                            "right": "node_0:value",
                        },
                    },
                ],
                "vmap": {
                    "mat": "node_2:value",
                },
            },
        ],
        "entry": "main",
    })
    # - main:
    # -x x+1
    # -x x+2
    #
    # - main:node_0
    # -2x 2(x+1)
    # -2x 2(x+2)
    #
    # - procedure:node_1
    # 4x*x-4(x+1)*x -4x*(x+1)+4(x+1)*(x+2)
    # 4x*x-4(x+2)*x -4x*(x+1)+4(x+2)*(x+2)
    #
    # - procedure:node_2
    # -4x 0
    # 0   12x+16
    #
    # - main:node_2
    # -4x+1 1
    # 1     12x+17
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue(TaskValueContainer({
                "value": create_tensor(np.array([
                    [-tix, tix + 1],
                    [-tix, tix + 2],
                ]), "float"),
            })),
            np.array([
                [1.0 - 4.0 * tix, 1.0],
                [1.0, 12.0 * tix + 17.0],
            ]),
        )
        for tix in range(20)
    ]
    time_start = time.monotonic()
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    try:
        config.run()
        for task_id, response, expected_result in wait_for_tasks(
                config, tasks, timeout=1.0):
            response_ok(response, no_warn=True)
            real_duration = time.monotonic() - time_start
            status = response["status"]
            result = response["result"]
            task_duration = response["duration"]
            retries = response["retries"]
            error = response["error"]
            assert status == TASK_STATUS_READY
            assert result is not None
            assert task_duration <= real_duration
            assert list(result["value"].shape) == [2, 2]
            assert retries == 0
            assert error is None
            np.testing.assert_allclose(
                as_numpy(result["value"]), expected_result)
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        emng = config.get_executor_manager()
        emng.release_all(timeout=1.0)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
