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


StrategyModule = TypedDict('StrategyModule', {
    "node": NodeStrategyModule,
    "queue": QueueStrategyModule,
})


LoggerDef = TypedDict('LoggerDef', {
    "listeners": list[EventListenerDef],
    "disable_events": list[str],
})


ConfigJSON = TypedDict('ConfigJSON', {
    "client_pool": ClientPoolModule,
    "data_store": DataStoreModule,
    "executor_manager": ExecutorManagerModule,
    "queue_pool": QueuePoolModule,
    "strategy": StrategyModule,
    "readonly_access": ReadonlyAccessModule,
    "logger": LoggerDef,
})


def load_config(
        exec_gen: Callable[[], ExecutorId],
        config_obj: ConfigJSON) -> Config:
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


def load_test(
        *,
        max_store_size: int = 1024 * 1024,
        parallelism: int = 0,
        batch_size: int = 5) -> Config:
    executor_manager: ExecutorManagerModule
    if parallelism > 0:
        executor_manager = {
            "name": "thread",
            "batch_size": batch_size,
            "parallelism": parallelism,
            "sleep_on_idle": 0.1,
        }
    else:
        executor_manager = {
            "name": "single",
            "batch_size": batch_size,
        }
    test_config: ConfigJSON = {
        "client_pool": {
            "name": "local",
        },
        "data_store": {
            "name": "local",
            "max_size": max_store_size,
        },
        "executor_manager": executor_manager,
        "queue_pool": {
            "name": "local",
            "check_assertions": True,
        },
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
