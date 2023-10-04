import numpy as np
import pytest

from scattermind.system.base import TaskId
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


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
def test_cop(base: list[list[float]], batch_size: int) -> None:
    shape = [len(base), len(base[0])]
    config = load_test(batch_size=batch_size)
    config.load_graph({
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
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue(TaskValueContainer({
                "value": create_tensor(np.array(base) * tix, "float"),
            })),
            np.array(base) * tix * 2.0,
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    config.run()
    for task_id, expected_result in tasks:
        response = config.get_response(task_id)
        response_ok(response, no_warn=True)
        assert response["status"] == TASK_STATUS_READY
        result = response["result"]
        assert result is not None
        assert list(result["value"].shape) == shape
        np.testing.assert_allclose(as_numpy(result["value"]), expected_result)
        assert config.get_status(task_id) == TASK_STATUS_DONE
        config.clear_task(task_id)
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None


@pytest.mark.parametrize("base", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
def test_cop_chain(base: list[list[float]], batch_size: int) -> None:
    shape = [len(base), len(base[0])]
    config = load_test(batch_size=batch_size)
    config.load_graph({
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
    for task_id, expected_result in tasks:
        response = config.get_response(task_id)
        response_ok(response, no_warn=True)
        assert response["status"] == TASK_STATUS_READY
        result = response["result"]
        assert result is not None
        assert list(result["value"].shape) == shape
        np.testing.assert_allclose(as_numpy(result["value"]), expected_result)
        assert config.get_status(task_id) == TASK_STATUS_DONE
        config.clear_task(task_id)
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None
