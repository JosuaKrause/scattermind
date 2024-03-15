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
"""Test graph caching."""
# import time
# from test.util import wait_for_tasks

# import numpy as np
import pytest

# from scattermind.system.base import TaskId
# from scattermind.system.client.client import TASK_MAX_RETRIES
from scattermind.system.base import set_debug_output_length
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace


# from scattermind.system.payload.values import TaskValueContainer
# from scattermind.system.response import (
#     response_ok,
#     TASK_STATUS_DONE,
#     TASK_STATUS_ERROR,
#     TASK_STATUS_READY,
#     TASK_STATUS_UNKNOWN,
#     TASK_STATUS_WAIT,
# )
# from scattermind.system.torch_util import as_numpy, create_tensor


@pytest.mark.parametrize("batch_size", [1, 5, 11, 20, 50])
@pytest.mark.parametrize("parallelism", [0, 1, 2, 3])
@pytest.mark.parametrize("is_redis", [False, True])
def test_entry_graph_cache(
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
                    "value_0": ("uint8", [None]),
                    "value_1": ("uint8", [None]),
                    "value_2": ("uint8", [None]),
                },
                "output_format": {
                    "value": ("uint8", [None]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-main-0",
                        },
                        "outs": {
                            "out": "node_1",
                        },
                        "vmap": {
                            "text": ":value_0",
                        },
                    },
                    {
                        "name": "node_1",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-main-1",
                        },
                        "outs": {
                            "out": "node_2",
                        },
                        "vmap": {
                            "text": ":value_1",
                        },
                    },
                    {
                        "name": "node_2",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-main-2",
                        },
                        "outs": {
                            "out": "node_3",
                        },
                        "vmap": {
                            "text": ":value_2",
                        },
                    },
                    {
                        "name": "node_3",
                        "kind": "call",
                        "args": {
                            "graph": "top",
                            "args": {
                                "value_0": ("uint8", [None]),
                            },
                            "ret": {
                                "value": ("uint8", [None]),
                            },
                        },
                        "outs": {
                            "out": "node_4",
                        },
                        "vmap": {
                            "value_0": "node_0:text",
                        },
                    },
                    {
                        "name": "node_4",
                        "kind": "call",
                        "args": {
                            "graph": "mid",
                            "args": {
                                "value_0": ("uint8", [None]),
                                "value_1": ("uint8", [None]),
                            },
                            "ret": {
                                "value": ("uint8", [None]),
                            },
                        },
                        "outs": {
                            "out": "node_5",
                        },
                        "vmap": {
                            "value_0": "node_1:text",
                            "value_1": "node_2:text",
                        },
                    },
                    {
                        "name": "node_5",
                        "kind": "str_concat",
                        "args": {
                            "delimiter": "-",
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "left": "node_3:value",
                            "right": "node_4:value",
                        },
                    },
                ],
                "vmap": {
                    "value": "node_5:value",
                },
            },
            {
                "name": "mid",
                "description": "mid level graph",
                "input": "node_0",
                "input_format": {
                    "value_0": ("uint8", [None]),
                    "value_1": ("uint8", [None]),
                },
                "output_format": {
                    "value": ("uint8", [None]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-mid-0",
                        },
                        "outs": {
                            "out": "node_1",
                        },
                        "vmap": {
                            "text": ":value_0",
                        },
                    },
                    {
                        "name": "node_1",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-mid-1",
                        },
                        "outs": {
                            "out": "node_2",
                        },
                        "vmap": {
                            "text": ":value_1",
                        },
                    },
                    {
                        "name": "node_2",
                        "kind": "call",
                        "args": {
                            "graph": "top",
                            "args": {
                                "value_0": ("uint8", [None]),
                            },
                            "ret": {
                                "value": ("uint8", [None]),
                            },
                        },
                        "outs": {
                            "out": "node_3",
                        },
                        "vmap": {
                            "value_0": "node_0:text",
                        },
                    },
                    {
                        "name": "node_3",
                        "kind": "str_concat",
                        "args": {
                            "delimiter": ":",
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "left": "node_2:value",
                            "right": "node_1:text",
                        },
                    },
                ],
                "vmap": {
                    "value": "node_3:value",
                },
            },
            {
                "name": "top",
                "description": "top level graph",
                "input": "node_0",
                "input_format": {
                    "value_0": ("uint8", [None]),
                },
                "output_format": {
                    "value": ("uint8", [None]),
                },
                "nodes": [
                    {
                        "name": "node_0",
                        "kind": "test_cache",
                        "args": {
                            "postfix": "-top",
                        },
                        "outs": {
                            "out": None,
                        },
                        "vmap": {
                            "text": ":value_0",
                        },
                    },
                ],
                "vmap": {
                    "value": "node_0:text",
                },
            },
        ],
        "entry": "main",
    })
    assert ns == GNamespace("main")
    # time_start = time.monotonic()
    # tasks: list[tuple[TaskId, bool]] = [
    #     (
    #         config.enqueue(
    #             ns,
    #             TaskValueContainer({
    #                 "value": create_tensor(
    #                     np.array([tix % 3 > 0]), dtype="bool"),
    #             })),
    #         tix % 3 > 0,
    #     )
    #     for tix in range(20)
    # ]
    # for task_id, _ in tasks:
    #     assert config.get_status(task_id) == TASK_STATUS_WAIT
    # try:
    #     config.run()
    #     for task_id, response, expected_result in wait_for_tasks(
    #             config, tasks):
    #         real_duration = time.monotonic() - time_start
    #         status = response["status"]
    #         task_ns = response["ns"]
    #         task_duration = response["duration"]
    #         result = response["result"]
    #         retries = response["retries"]
    #         error = response["error"]
    #         assert task_duration <= real_duration
    #         assert task_ns == ns
    #         if expected_result:
    #             response_ok(response, no_warn=True)
    #             assert status == TASK_STATUS_READY
    #             assert result is not None
    #             assert list(result["value"].shape) == [1]
    #             assert retries == 0
    #             assert error is None
    #             np.testing.assert_allclose(
    #                 as_numpy(result["value"]), np.array([expected_result]))
    #             assert config.get_status(task_id) == TASK_STATUS_DONE
    #         else:
    #             assert status == TASK_STATUS_ERROR
    #             assert result is None
    #             assert error is not None
    #             assert error["code"] == "general_exception"
    #             ectx = error["ctx"]
    #             assert ectx["node_name"] is not None
    #             assert ectx["node_name"].get() == "node_1"
    #             assert ectx["graph_name"] is not None
    #             assert ectx["graph_name"].get_namespace().get() == "main"
    #             assert ectx["graph_name"].get_name().get() == "main"
    #             assert error["message"].find("value was not true") >= 0
    #             tback = error["traceback"]
    #             assert "Traceback" in tback[0]
    #             assert "line" in tback[-2]
    #             assert "ValueError: value was not true" in tback[-1]
    #             with pytest.raises(ValueError, match=r"value was not true"):
    #                 response_ok(response, no_warn=True)
    #             assert retries == TASK_MAX_RETRIES
    #         config.clear_task(task_id)
    #         assert config.get_namespace(task_id) is None
    #         assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
    #         assert config.get_result(task_id) is None
    # finally:
    #     print("TEST TEARDOWN!")
    #     emng = config.get_executor_manager()
    #     emng.release_all(timeout=1.0)
    #     if emng.any_active():
    #         raise ValueError("threads did not shut down in time")
