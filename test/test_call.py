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
"""Test calling subgraphs."""
import time
from test.util import wait_for_tasks

import numpy as np
import pytest

from scattermind.system.base import set_debug_output_length, TaskId
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace
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
@pytest.mark.parametrize("is_redis", [False, True])
def test_simple_call(
        batch_size: int, parallelism: int, is_redis: bool) -> None:
    """
    Test a simple subgraph call.

    Args:
        batch_size (int): The batch size.
        parallelism (int): The parallelism of the executor.
        is_redis (bool): Whether to use redis.
    """
    set_debug_output_length(7)
    config = load_test(
        batch_size=batch_size, parallelism=parallelism, is_redis=is_redis)
    writer = config.get_roa_writer()
    cpath = "procedure_const_0"
    with writer.open_write(cpath) as fout:
        constant_0 = fout.as_data_str(fout.write_tensor(
            create_tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), dtype="float")))
    ns = config.load_graph({
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
    assert ns == GNamespace("main")
    time_start = time.monotonic()
    tasks: list[tuple[TaskId, np.ndarray]] = [
        (
            config.enqueue_task(
                "main",
                {
                    "value": np.array([
                        [-tix, tix + 1],
                        [-tix, tix + 2],
                    ], dtype=np.float32),
                }),
            np.array([
                [1.0 - 4.0 * tix, 1.0],
                [1.0, 12.0 * tix + 17.0],
            ]),
        )
        for tix in range(20)
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    try:
        config.run()
        for task_id, response, expected_result in wait_for_tasks(
                config, tasks):
            response_ok(response, no_warn=True)
            real_duration = time.monotonic() - time_start
            status = response["status"]
            task_ns = response["ns"]
            result = response["result"]
            task_duration = response["duration"]
            retries = response["retries"]
            error = response["error"]
            assert status == TASK_STATUS_READY
            assert task_ns == ns
            assert result is not None
            assert task_duration <= real_duration
            assert list(result["value"].shape) == [2, 2]
            assert retries == 0
            assert error is None
            np.testing.assert_allclose(
                as_numpy(result["value"]), expected_result)
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        emng = config.get_executor_manager()
        emng.release_all(timeout=1.0)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
