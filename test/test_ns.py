"""Testing namespaces."""
import numpy as np
import pytest

from scattermind.system.base import TaskId
from scattermind.system.config.loader import load_test
from scattermind.system.response import (
    response_ok,
    TASK_STATUS_DONE,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import as_numpy


@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("is_redis", [False, True])
def test_ns(batch_size: int, is_redis: bool) -> None:
    """
    Test loading multiple graphs.

    Args:
        batch_size (int): The batch size for processing.
        is_redis (bool): Whether to use redis.
    """
    config = load_test(batch_size=batch_size, is_redis=is_redis)
    config.load_graph({
        "graphs": [
            {
                "name": "main",
                "description": f"batch_size={batch_size}",
                "input": "node",
                "input_format": {
                    "value": ("float", [1]),
                },
                "output_format": {
                    "value": ("float", [1]),
                },
                "nodes": [
                    {
                        "name": "node",
                        "kind": "constant_op",
                        "args": {
                            "op": "mul",
                            "const": 2,
                            "input": ("float", [1]),
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
        "entry": "main",
        "ns": "graph_1",
    })
    config.load_graph({
        "graphs": [
            {
                "name": "main",
                "description": f"batch_size={batch_size}",
                "input": "node",
                "input_format": {
                    "value": ("float", [1]),
                },
                "output_format": {
                    "value": ("float", [1]),
                },
                "nodes": [
                    {
                        "name": "node",
                        "kind": "constant_op",
                        "args": {
                            "op": "mul",
                            "const": 3,
                            "input": ("float", [1]),
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
        "entry": "main",
        "ns": "graph_2",
    })
    tasks: list[tuple[TaskId, np.ndarray, str]] = [
        (
            config.enqueue_task(
                f"graph_{gix}",
                {
                    "value": np.array([tix], dtype=np.float32),
                }),
            np.array([tix * 2.0 if gix == 1 else tix * 3.0]),
            f"graph_{gix}",
        )
        for gix in [1, 2]
        for tix in range(20)
    ]
    for task_id, _, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    config.run()
    for task_id, expected_result, ns in tasks:
        response = config.get_response(task_id)
        response_ok(response, no_warn=True)
        assert response["status"] == TASK_STATUS_READY
        result = response["result"]
        assert result is not None
        assert list(result["value"].shape) == [1]
        np.testing.assert_allclose(as_numpy(result["value"]), expected_result)
        assert config.get_status(task_id) == TASK_STATUS_DONE
        assert config.get_namespace(task_id).get() == ns
        config.clear_task(task_id)
        assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
        assert config.get_result(task_id) is None
