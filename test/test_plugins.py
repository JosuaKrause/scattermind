import os
import shutil
import uuid
from collections.abc import Callable
from typing import Any, cast, TypeVar

import pytest

from scatterbrain.system.base import ExecutorId, once_test_executors
from scatterbrain.system.client.client import ClientPool
from scatterbrain.system.client.loader import (
    ClientPoolModule,
    load_client_pool,
)
from scatterbrain.system.executor.executor import ExecutorManager
from scatterbrain.system.executor.loader import (
    ExecutorManagerModule,
    load_executor_manager,
)
from scatterbrain.system.graph.loader import load_node
from scatterbrain.system.graph.node import Node
from scatterbrain.system.payload.data import DataStore
from scatterbrain.system.payload.loader import DataStoreModule, load_store
from scatterbrain.system.plugins import load_plugin
from scatterbrain.system.queue.loader import load_queue_pool, QueuePoolModule
from scatterbrain.system.queue.queue import QueuePool
from scatterbrain.system.queue.strategy.loader import (
    load_node_strategy,
    load_queue_strategy,
    NodeStrategyModule,
    QueueStrategyModule,
)
from scatterbrain.system.queue.strategy.strategy import (
    NodeStrategy,
    QueueStrategy,
)
from scatterbrain.system.readonly.access import ReadonlyAccess
from scatterbrain.system.readonly.loader import (
    load_readonly_access,
    ReadonlyAccessModule,
)


T = TypeVar('T')


def test_plugins() -> None:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    plugins_base = os.path.normpath(os.path.join(root, "plugins/test/"))
    test_id = f"test_{uuid.uuid4().hex}"
    plugins_folder = os.path.normpath(
        os.path.join(plugins_base, f"{test_id}/"))
    os.makedirs(plugins_folder, exist_ok=True)
    base_init = os.path.join(plugins_base, "__init__.py")
    if not os.path.exists(base_init):
        with open(base_init, "wb") as fout:
            fout.flush()
    with open(os.path.join(plugins_folder, "__init__.py"), "wb") as fout:
        fout.flush()
    test_ix = 0
    tests = []

    def do_test(
            base: type[T],
            name: str,
            load_fun: Callable[[dict[str, Any]], T],
            args: dict[str, Any]) -> None:
        nonlocal test_ix

        src_file = os.path.join(root, f"src/{name.replace('.', '/')}.py")
        dest_file = os.path.join(plugins_folder, f"test_{test_ix}.py")
        shutil.copy(src_file, dest_file)
        args["name"] = f"plugins.test.{test_id}.test_{test_ix}"
        tests.append((base, load_fun, args))
        test_ix += 1

    def run_tests() -> None:
        for base, load_fun, args in tests:
            success = False
            try:
                plugin = load_fun(args)
                assert isinstance(plugin, base)
                success = True
            finally:
                if not success:
                    path = plugins_folder
                    for _ in range(4):
                        print(f"folder: {path}")
                        for fname in sorted(os.listdir(path)):
                            print(fname)
                        path = os.path.normpath(os.path.dirname(path))

    do_test(
        ClientPool,
        "scatterbrain.system.client.local",
        lambda args: load_client_pool(cast(ClientPoolModule, args)), {})
    do_test(
        ExecutorManager,
        "scatterbrain.system.executor.single",
        lambda args: load_executor_manager(
            once_test_executors([ExecutorId.for_test()]),
            cast(ExecutorManagerModule, args)),
        {
            "batch_size": 5,
        })
    do_test(
        Node,
        "scatterbrain.system.graph.nodes.mat_square",
        lambda args: load_node(None, args["name"], None),  # type: ignore
        {})
    do_test(
        DataStore,
        "scatterbrain.system.payload.local",
        lambda args: load_store(cast(DataStoreModule, args)),
        {
            "max_size": 1000,
        })
    do_test(
        QueuePool,
        "scatterbrain.system.queue.local",
        lambda args: load_queue_pool(cast(QueuePoolModule, args)),
        {
            "check_assertions": False,
        })
    do_test(
        NodeStrategy,
        "scatterbrain.system.queue.strategy.node.simple",
        lambda args: load_node_strategy(cast(NodeStrategyModule, args)),
        {})
    do_test(
        QueueStrategy,
        "scatterbrain.system.queue.strategy.queue.simple",
        lambda args: load_queue_strategy(cast(QueueStrategyModule, args)),
        {})
    do_test(
        ReadonlyAccess,
        "scatterbrain.system.readonly.ram",
        lambda args: load_readonly_access(cast(ReadonlyAccessModule, args)),
        {})

    run_tests()


def test_invalid_plugins() -> None:
    with pytest.raises(ValueError, match=r"ambiguous or missing plugin"):
        load_plugin(ClientPool, "plugins.test")
