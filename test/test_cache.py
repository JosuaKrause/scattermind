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
    ResponseObject,
    TASK_STATUS_DONE,
    TASK_STATUS_READY,
    TASK_STATUS_UNKNOWN,
)
from scattermind.system.torch_util import str_to_tensor, tensor_to_str
from scattermind.system.util import get_short_hash


@pytest.mark.parametrize("parallelism", [1, 2, 4, -1])
@pytest.mark.parametrize("is_redis", [False, True])
@pytest.mark.parametrize("is_cache", [False, True])
@pytest.mark.parametrize("cache_main", [False, True])
@pytest.mark.parametrize("cache_mid", [False])  # FIXME: enable when ready
@pytest.mark.parametrize("cache_top", [False])  # FIXME: enable when ready
def test_graph_cache(
        parallelism: int,
        is_redis: bool,
        is_cache: bool,
        cache_main: bool,
        cache_mid: bool,
        cache_top: bool) -> None:
    """
    Test for graph caching.

    Args:
        parallelism (int): The parallelism. Cannot use single executor since
            it terminates upon completion.
        is_redis (bool): Whether to use redis.
        is_cache (bool): Whether to cache at all.
        cache_main (bool): Whether to cache the main graph.
        cache_mid (bool): Whether to cache the mid graph.
        cache_top (bool): Whether to cache the top graph.
    """
    # NOTE: batch_size can only be one to guarantee caching comes into effect
    set_debug_output_length(7)
    config = load_test(
        parallelism=parallelism,
        is_redis=is_redis,
        no_cache=not is_cache)
    desc = (
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
    time_start = time.monotonic()
    inputs: list[tuple[str, str, str]] = [
        ("a", "b", "c"),
        ("d", "e", "f"),
        ("a", "b", "c"),
        ("a", "b", "c"),
        ("a", "b", "c"),
        ("d", "g", "h"),
        ("d", "g", "h"),
        ("i", "e", "k"),
        ("i", "e", "k"),
        ("i", "k", "h"),
        ("e", "i", "k"),
        ("e", "i", "k"),
        ("a", "e", "h"),
        ("a", "e", "h"),
    ]
    cache_main_obj: dict[tuple[str, str, str], str] = {}
    cache_mid_obj: dict[tuple[str, str], str] = {}
    cache_top_obj: dict[str, str] = {}
    rep_main_0: set[str] = set()
    rep_main_1: set[str] = set()
    rep_main_2: set[str] = set()
    rep_mid_0: set[str] = set()
    rep_mid_1: set[str] = set()
    rep_top: set[str] = set()
    pf_top = "-top"
    pf_mid_0 = "-mid-0"
    pf_mid_1 = "-mid-1"
    pf_main_0 = "-main-0"
    pf_main_1 = "-main-1"
    pf_main_2 = "-main-2"

    def top_expected(top: str) -> str:
        res_top = cache_top_obj.get(top)
        if res_top is not None:
            return res_top
        if top in rep_top:
            val = f"{top}{pf_top}"
        else:
            rep_top.add(top)
            val = top
        if is_cache and cache_top:
            cache_top_obj[top] = val
        return val

    def mid_expected(mid: tuple[str, str]) -> str:
        res_mid = cache_mid_obj.get(mid)
        if res_mid is not None:
            return res_mid
        val_0, val_1 = mid
        if val_0 in rep_mid_0:
            val_0 = f"{val_0}{pf_mid_0}"
        else:
            rep_mid_0.add(val_0)
        if val_1 in rep_mid_1:
            val_1 = f"{val_1}{pf_mid_1}"
        else:
            rep_mid_1.add(val_1)
        val_0 = top_expected(val_0)
        val = f"{val_0}:{val_1}"
        if is_cache and cache_mid:
            cache_mid_obj[mid] = val
        return val

    def compute_expected(input_tup: tuple[str, str, str]) -> str:
        res_main = cache_main_obj.get(input_tup)
        if res_main is not None:
            return res_main
        val_0, val_1, val_2 = input_tup
        if val_0 in rep_main_0:
            val_0 = f"{val_0}{pf_main_0}"
        else:
            rep_main_0.add(val_0)
        if val_1 in rep_main_1:
            val_1 = f"{val_1}{pf_main_1}"
        else:
            rep_main_1.add(val_1)
        if val_2 in rep_main_2:
            val_2 = f"{val_2}{pf_main_2}"
        else:
            rep_main_2.add(val_2)
        val_0 = top_expected(val_0)
        val_mid = mid_expected((val_1, val_2))
        val = f"{val_0}-{val_mid}"
        if is_cache and cache_main:
            cache_main_obj[input_tup] = val
        return val

    try:
        config.run(force_no_block=True)
        print(
            f"SETTING: is_cache={is_cache} cache_main={cache_main} "
            f"cache_mid={cache_mid} cache_top={cache_top}")
        for cur_input in inputs:
            expected = compute_expected(cur_input)
            print(f"EXECUTING TASK: {cur_input} with {expected}")
            task_id = config.enqueue(
                ns,
                TaskValueContainer({
                    "value_0": str_to_tensor(cur_input[0]),
                    "value_1": str_to_tensor(cur_input[1]),
                    "value_2": str_to_tensor(cur_input[2]),
                }))
            cur_res: list[tuple[TaskId, ResponseObject, None]] = \
                list(wait_for_tasks(config, [(task_id, None)]))
            response: ResponseObject = cur_res[0][1]
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
            assert tensor_to_str(result["value"]) == expected
            assert config.get_status(task_id) == TASK_STATUS_DONE
            config.clear_task(task_id)
            assert config.get_namespace(task_id) is None
            assert config.get_status(task_id) == TASK_STATUS_UNKNOWN
            assert config.get_result(task_id) is None
    finally:
        print("TEST TEARDOWN!")
        emng = config.get_executor_manager()
        emng.release_all(timeout=0.1)
        if emng.any_active():
            raise ValueError("threads did not shut down in time")
