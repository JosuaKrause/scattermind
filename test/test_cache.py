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
import time
from test.util import wait_for_tasks

import pytest

from scattermind.system.base import set_debug_output_length, TaskId
from scattermind.system.config.loader import load_test
from scattermind.system.names import GNamespace
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    response_ok,
    TASK_STATUS_DONE,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
    TASK_STATUS_WAIT,
)
from scattermind.system.torch_util import str_to_tensor, tensor_to_str
from scattermind.system.util import get_short_hash


@pytest.mark.parametrize("batch_size", [1, 2, 8, 14, 20])
@pytest.mark.parametrize("parallelism", [0, 1])
@pytest.mark.parametrize("is_redis", [False])
@pytest.mark.parametrize("is_cache", [False, True])
@pytest.mark.parametrize("cache_main", [False, True])
@pytest.mark.parametrize("cache_mid", [False])  # FIXME: enable when ready
@pytest.mark.parametrize("cache_top", [False])  # FIXME: enable when ready
def test_graph_cache(
        batch_size: int,
        parallelism: int,
        is_redis: bool,
        is_cache: bool,
        cache_main: bool,
        cache_mid: bool,
        cache_top: bool) -> None:
    """
    Test for graph caching.

    Args:
        batch_size (int): The batch size.
        parallelism (int): The parallelism. We can only use single threads as
            otherwise caching behaviors would be unpredictable.
        is_redis (bool): Whether to use redis.
        is_cache (bool): Whether to cache at all.
        cache_main (bool): Whether to cache the main graph.
        cache_mid (bool): Whether to cache the mid graph.
        cache_top (bool): Whether to cache the top graph.
    """
    # FIXME: make is_redis=True tests work
    set_debug_output_length(7)
    config = load_test(
        batch_size=batch_size,
        parallelism=parallelism,
        is_redis=is_redis,
        no_cache=not is_cache)
    desc = (
        f"batch_size={batch_size};"
        f"parallelism={parallelism};"
        f"is_redis={is_redis};"
        f"is_cache={is_cache}")
    seed = get_short_hash(desc)
    ns = config.load_graph({
        "graphs": [
            {
                "name": "main",
                "description": desc,
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
                            "seed": seed,
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
                            "seed": seed,
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
                            "seed": seed,
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
                "cache": cache_main,
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
                            "seed": seed,
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
                            "seed": seed,
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
                "cache": cache_mid,
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
                            "seed": seed,
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
                "cache": cache_top,
            },
        ],
        "entry": "main",
    })
    assert ns == GNamespace("main")
    pf_top = "" if cache_top and is_cache else "-top"
    pf_mid_0 = "" if cache_mid and is_cache else "-mid-0"
    pf_mid_1 = "" if cache_mid and is_cache else "-mid-1"
    pf_main_0 = "" if cache_main and is_cache else "-main-0"
    pf_main_1 = "" if cache_main and is_cache else "-main-1"
    pf_main_2 = "" if cache_main and is_cache else "-main-2"
    time_start = time.monotonic()
    # NOTE: we need to treat every batch_size >= len(inputs) differently
    # as the caching behavior changes
    inputs: list[tuple[tuple[str, str, str], str]] = [
        (("a", "b", "c"), "a-b:c"),
        (("d", "e", "f"), "d-e:f" if batch_size < 14 else f"d-e{pf_top}:f"),
        (("a", "b", "c"), f"a{pf_main_0}-b{pf_main_1}:c{pf_main_2}"),
        (
            ("a", "b", "c"),
            f"a{pf_main_0}{pf_top}-"
            f"b{pf_main_1}{pf_mid_0}:c{pf_main_2}{pf_mid_1}",
        ),
        (
            ("a", "b", "c"),
            f"a{pf_main_0}{pf_top}-"
            f"b{pf_main_1}{pf_mid_0}{pf_top}:c{pf_main_2}{pf_mid_1}",
        ),
        (("d", "g", "h"), f"d{pf_main_0}-g:h"),
        (("d", "g", "h"), f"d{pf_main_0}{pf_top}-g{pf_main_1}:h{pf_main_2}"),
        (("i", "e", "k"), f"i-e{pf_main_1}:k"),
        (("i", "e", "k"), f"i{pf_main_0}-e{pf_main_1}{pf_mid_0}:k{pf_main_2}"),
        (("i", "k", "h"), f"i{pf_main_0}{pf_top}-k:h{pf_main_2}{pf_mid_1}"),
        (
            ("e", "i", "k"),
            f"e{pf_top}-i{pf_top}:k{pf_main_2}{pf_mid_1}"
            if batch_size < 14 else
            f"e-i{pf_top}:k{pf_main_2}{pf_mid_1}",
        ),
        (
            ("e", "i", "k"),
            f"e{pf_main_0}-i{pf_main_1}:k{pf_main_2}{pf_mid_1}",
        ),
        (
            ("a", "e", "h"),
            f"a{pf_main_0}{pf_top}-"
            f"e{pf_main_1}{pf_mid_0}{pf_top}:h{pf_main_2}{pf_mid_1}",
        ),
        (
            ("a", "e", "h"),
            f"a{pf_main_0}{pf_top}-"
            f"e{pf_main_1}{pf_mid_0}{pf_top}:h{pf_main_2}{pf_mid_1}",
        ),
    ]
    tasks: list[tuple[TaskId, str]] = [
        (
            config.enqueue(
                ns,
                TaskValueContainer({
                    "value_0": str_to_tensor(cur_input[0]),
                    "value_1": str_to_tensor(cur_input[1]),
                    "value_2": str_to_tensor(cur_input[2]),
                })),
            expected,
        )
        for cur_input, expected in inputs
    ]
    for task_id, _ in tasks:
        assert config.get_status(task_id) == TASK_STATUS_WAIT
    try:
        config.run()
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
            response_ok(response, no_warn=True)
            assert status == TASK_STATUS_READY
            assert result is not None
            assert len(result["value"].shape) == 1
            assert retries == 0
            assert error is None
            assert tensor_to_str(result["value"]) == expected_result
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        print("TEST TEARDOWN!")
        emng = config.get_executor_manager()
        emng.release_all(timeout=1.0)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
