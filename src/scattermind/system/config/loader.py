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
"""Loads a configuration from a JSON file."""
from collections.abc import Callable
from typing import NotRequired, TypedDict

from scattermind.system.base import ExecutorId, once_test_executors
from scattermind.system.cache.loader import GraphCacheModule, load_graph_cache
from scattermind.system.client.loader import ClientPoolModule, load_client_pool
from scattermind.system.config.config import Config
from scattermind.system.executor.loader import (
    ExecutorManagerModule,
    load_executor_manager,
)
from scattermind.system.logger.loader import (
    EventListenerDef,
    load_event_listener,
)
from scattermind.system.logger.log import EventStream
from scattermind.system.payload.loader import DataStoreModule, load_store
from scattermind.system.queue.loader import load_queue_pool, QueuePoolModule
from scattermind.system.queue.strategy.loader import (
    load_node_strategy,
    load_queue_strategy,
    NodeStrategyModule,
    QueueStrategyModule,
)
from scattermind.system.readonly.loader import (
    load_readonly_access,
    ReadonlyAccessModule,
)
from scattermind.system.redis_util import get_test_config


HealthCheck = TypedDict('HealthCheck', {
    "address_in": str,
    "address_out": str,
    "port": int,
})
"""Address at which a healthcheck is exposed."""


StrategyModule = TypedDict('StrategyModule', {
    "node": NodeStrategyModule,
    "queue": QueueStrategyModule,
})
"""Module for selecting the node and the queue strategy."""


LoggerDef = TypedDict('LoggerDef', {
    "listeners": list[EventListenerDef],
    "disable_events": list[str],
})
"""Define the logger. `listeners` is a list of all listeners that process the
logs. `disable_events` is list of patterns to filter or include certain log
types."""


ConfigJSON = TypedDict('ConfigJSON', {
    "client_pool": ClientPoolModule,
    "data_store": DataStoreModule,
    "executor_manager": ExecutorManagerModule,
    "queue_pool": QueuePoolModule,
    "graph_cache": GraphCacheModule,
    "strategy": StrategyModule,
    "readonly_access": ReadonlyAccessModule,
    "logger": LoggerDef,
    "healthcheck": NotRequired[HealthCheck],
})
"""The configuration JSON."""


def load_config(
        exec_gen: Callable[[], ExecutorId],
        config_obj: ConfigJSON) -> Config:
    """
    Load a configuration from a JSON.

    Args:
        exec_gen (Callable[[], ExecutorId]): The executor generator function.
        config_obj (ConfigJSON): The configuration JSON.

    Returns:
        Config: The configuration.
    """
    config = Config()
    logger = EventStream()
    logger_obj = config_obj["logger"]
    for listener_def in logger_obj["listeners"]:
        logger.add_listener(
            load_event_listener(listener_def, logger_obj["disable_events"]))
    config.set_logger(logger)
    config.set_data_store(load_store(config_obj["data_store"]))
    config.set_executor_manager(
        load_executor_manager(exec_gen, config_obj["executor_manager"]))
    config.set_queue_pool(load_queue_pool(config_obj["queue_pool"]))
    config.set_client_pool(load_client_pool(config_obj["client_pool"]))
    config.set_graph_cache(load_graph_cache(config_obj["graph_cache"]))
    strategy_obj = config_obj["strategy"]
    config.set_node_strategy(load_node_strategy(strategy_obj["node"]))
    config.set_queue_strategy(load_queue_strategy(strategy_obj["queue"]))
    config.set_readonly_access(
        load_readonly_access(config_obj["readonly_access"]))
    hc = config_obj.get("healthcheck")
    if hc is not None:
        config.set_healthcheck(hc["address_in"], hc["address_out"], hc["port"])
    return config


def load_as_api(config_obj: ConfigJSON) -> Config:
    """
    Load a configuration from a JSON without loading an executor manager.

    Args:
        config_obj (ConfigJSON): The configuration JSON.

    Returns:
        Config: The API configuration.
    """
    config = Config()
    logger = EventStream()
    logger_obj = config_obj["logger"]
    for listener_def in logger_obj["listeners"]:
        logger.add_listener(
            load_event_listener(listener_def, logger_obj["disable_events"]))
    config.set_logger(logger)
    config.set_data_store(load_store(config_obj["data_store"]))
    config.set_queue_pool(load_queue_pool(config_obj["queue_pool"]))
    config.set_client_pool(load_client_pool(config_obj["client_pool"]))
    config.set_graph_cache(load_graph_cache(config_obj["graph_cache"]))
    strategy_obj = config_obj["strategy"]
    config.set_node_strategy(load_node_strategy(strategy_obj["node"]))
    config.set_queue_strategy(load_queue_strategy(strategy_obj["queue"]))
    config.set_readonly_access(
        load_readonly_access(config_obj["readonly_access"]))
    hc = config_obj.get("healthcheck")
    if hc is not None:
        config.set_healthcheck(hc["address_in"], hc["address_out"], hc["port"])
    return config


def load_test(
        *,
        is_redis: bool,
        max_store_size: int = 1024 * 1024,
        parallelism: int = 0,
        batch_size: int = 5,
        no_cache: bool = True) -> Config:
    """
    Load a configuration for unit tests.

    Args:
        is_redis (bool): Whether the test uses redis.
        max_store_size (int, optional): The maximum payload data store size
            for local stores. Defaults to 1024*1024.
        parallelism (int, optional): Whether use multiple executor managers
            via threads. If 0 a single executor is used. If -1 the redis
            executor is used. Defaults to 0.
        batch_size (int, optional): The batch size defining the number of
            tasks that get computed together. Defaults to 5.
        no_cache (bool, optional): Whether to use no caching. Defaults to True.

    Returns:
        Config: The configuration.
    """
    executor_manager: ExecutorManagerModule
    client_pool: ClientPoolModule
    data_store: DataStoreModule
    queue_pool: QueuePoolModule
    graph_cache: GraphCacheModule
    if parallelism > 0:
        executor_manager = {
            "name": "thread",
            "batch_size": batch_size,
            "parallelism": parallelism,
            "sleep_on_idle": 0.01,
            "reclaim_sleep": 60.0,
        }
    elif parallelism == -1:
        executor_manager = {
            "name": "redis",
            "batch_size": batch_size,
            "sleep_on_idle": 0.01,
            "reclaim_sleep": 60.0,
            "heartbeat_time": 1.0,
            "cfg": get_test_config(),
        }
    elif parallelism == 0:
        executor_manager = {
            "name": "single",
            "batch_size": batch_size,
        }
    else:
        raise ValueError(f"invalid parallelism: {parallelism}")
    if is_redis:
        client_pool = {
            "name": "redis",
            "cfg": get_test_config(),
        }
        data_store = {
            "name": "redis",
            "cfg": get_test_config(),
            "mode": "size",
        }
        queue_pool = {
            "name": "redis",
            "cfg": get_test_config(),
            "check_assertions": True,
        }
    else:
        client_pool = {
            "name": "local",
        }
        data_store = {
            "name": "local",
            "max_size": max_store_size,
        }
        queue_pool = {
            "name": "local",
            "check_assertions": True,
        }
    if no_cache:
        graph_cache = {
            "name": "nocache",
        }
    else:
        graph_cache = {
            "name": "redis",
            "cfg": get_test_config(),
        }
    test_config: ConfigJSON = {
        "client_pool": client_pool,
        "data_store": data_store,
        "executor_manager": executor_manager,
        "queue_pool": queue_pool,
        "graph_cache": graph_cache,
        "strategy": {
            "node": {
                "name": "simple",
            },
            "queue": {
                "name": "simple",
            },
        },
        "readonly_access": {
            "name": "ram",
            "scratch": "invalid",
        },
        "logger": {
            "listeners": [
                {
                    "name": "stdout",
                    "show_debug": True,
                },
            ],
            "disable_events": [
                "measure",
            ],
        },
    }
    execs = [ExecutorId.for_test()]
    for _ in range(max(0, parallelism - 1)):
        execs.append(ExecutorId.create())
    return load_config(once_test_executors(execs), test_config)
