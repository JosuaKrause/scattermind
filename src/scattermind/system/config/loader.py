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
"""Loads a configuration from a JSON file."""
from collections.abc import Callable
from typing import TypedDict

from scattermind.system.base import ExecutorId, once_test_executors
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
    "strategy": StrategyModule,
    "readonly_access": ReadonlyAccessModule,
    "logger": LoggerDef,
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
    strategy_obj = config_obj["strategy"]
    config.set_node_strategy(load_node_strategy(strategy_obj["node"]))
    config.set_queue_strategy(load_queue_strategy(strategy_obj["queue"]))
    config.set_readonly_access(
        load_readonly_access(config_obj["readonly_access"]))
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
    strategy_obj = config_obj["strategy"]
    config.set_node_strategy(load_node_strategy(strategy_obj["node"]))
    config.set_queue_strategy(load_queue_strategy(strategy_obj["queue"]))
    config.set_readonly_access(
        load_readonly_access(config_obj["readonly_access"]))
    return config


def load_test(
        *,
        is_redis: bool,
        max_store_size: int = 1024 * 1024,
        parallelism: int = 0,
        batch_size: int = 5) -> Config:
    """
    Load a configuration for unit tests.

    Args:
        is_redis (bool): Whether the test uses redis.
        max_store_size (int, optional): The maximum payload data store size
            for local stores. Defaults to 1024*1024.
        parallelism (int, optional): Whether use multiple executor managers
            via threads. Defaults to 0.
        batch_size (int, optional): The batch size defining the number of
            tasks that get computed together. Defaults to 5.

    Returns:
        Config: The configuration.
    """
    executor_manager: ExecutorManagerModule
    client_pool: ClientPoolModule
    data_store: DataStoreModule
    queue_pool: QueuePoolModule
    if parallelism > 0:
        executor_manager = {
            "name": "thread",
            "batch_size": batch_size,
            "parallelism": parallelism,
            "sleep_on_idle": 0.1,
            "reclaim_sleep": 60.0,
        }
    else:
        executor_manager = {
            "name": "single",
            "batch_size": batch_size,
        }
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
    test_config: ConfigJSON = {
        "client_pool": client_pool,
        "data_store": data_store,
        "executor_manager": executor_manager,
        "queue_pool": queue_pool,
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
