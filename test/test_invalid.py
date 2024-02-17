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
"""Test invalid configurations."""
import pytest

from scattermind.system.base import (
    DataId,
    ExecutorId,
    GraphId,
    NodeId,
    once_test_executors,
    QueueId,
    TaskId,
)
from scattermind.system.client.client import ComputeTask
from scattermind.system.client.loader import load_client_pool
from scattermind.system.config.config import Config
from scattermind.system.config.loader import load_test
from scattermind.system.executor.loader import load_executor_manager
from scattermind.system.graph.graph import Graph
from scattermind.system.graph.loader import load_node
from scattermind.system.info import DataFormat, DataInfo
from scattermind.system.names import (
    GName,
    GNamespace,
    NName,
    QualifiedGraphName,
    QualifiedName,
)
from scattermind.system.payload.loader import load_store
from scattermind.system.payload.local import LocalDataId
from scattermind.system.payload.values import DataContainer, TaskValueContainer
from scattermind.system.queue.loader import load_queue_pool
from scattermind.system.queue.strategy.loader import (
    load_node_strategy,
    load_queue_strategy,
)
from scattermind.system.readonly.loader import load_readonly_access


def test_loaders() -> None:
    """Test loading modules with invalid configurations."""
    with pytest.raises(ValueError, match=r"unknown node strategy"):
        load_node_strategy({"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown queue strategy"):
        load_queue_strategy({"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown queue pool"):
        load_queue_pool({"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown data store"):
        load_store({"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown node"):
        load_node(None, "foo", None)  # type: ignore
    with pytest.raises(
            ModuleNotFoundError, match=r"No module named .*plugins\.test.*"):
        load_node(None, "plugins.test.foo", None)  # type: ignore
    with pytest.raises(ValueError, match=r"unknown executor manager"):
        load_executor_manager(None, {"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown client pool"):
        load_client_pool({"name": "foo"})  # type: ignore
    with pytest.raises(ValueError, match=r"unknown readonly access"):
        load_readonly_access({"name": "foo"})  # type: ignore

    exec_gen = once_test_executors([ExecutorId.for_test()])
    assert exec_gen() is ExecutorId.for_test()
    with pytest.raises(
            ValueError, match=r"generator can only produce each value once"):
        exec_gen()


def test_queue_pool() -> None:
    """Test invalid configurations and graphs."""
    config = Config()
    ns = GNamespace("test")

    with pytest.raises(ValueError, match=r"graph .* not initialized"):
        config.get_graph(ns)
    graph = Graph(ns)
    gname = QualifiedGraphName(ns, GName("test"))
    graph_id = GraphId.create(gname)
    config.add_graph(graph)
    with pytest.raises(ValueError, match=r"graph .* already added"):
        config.add_graph(Graph(ns))
    assert graph is config.get_graph(ns)

    with pytest.raises(ValueError, match=r"executor manager not initialized"):
        config.get_executor_manager()
    executor_manager = load_executor_manager(
        ExecutorId.create,
        {
            "name": "single",
            "batch_size": 5,
        })
    config.set_executor_manager(executor_manager)
    with pytest.raises(
            ValueError, match=r"executor manager already initialized"):
        config.set_executor_manager(load_executor_manager(
            ExecutorId.create,
            {
                "name": "thread",
                "batch_size": 10,
                "parallelism": 1,
                "sleep_on_idle": 0.1,
                "reclaim_sleep": 60.0,
            }))

    with pytest.raises(ValueError, match=r"store not initialized"):
        config.get_data_store()
    store = load_store(
        {
            "name": "local",
            "max_size": 1000,
        })
    config.set_data_store(store)
    with pytest.raises(ValueError, match=r"store already initialized"):
        config.set_data_store(load_store(
            {
                "name": "local",
                "max_size": 2000,
            }))

    with pytest.raises(
            ValueError,
            match=r"readonly access needs to be initialized first"):
        config.get_readonly_access()
    roa = load_readonly_access({"name": "ram"})
    config.set_readonly_access(roa)
    with pytest.raises(
            ValueError, match=r"readonly access already initialized"):
        config.set_readonly_access(load_readonly_access({"name": "ram"}))

    with pytest.raises(
            ValueError, match=r"queue pool needs to be initialized first"):
        config.get_queue_pool()
    queue_pool = load_queue_pool({"name": "local"})
    config.set_queue_pool(queue_pool)
    with pytest.raises(ValueError, match=r"queue pool already initialized"):
        config.set_queue_pool(load_queue_pool({"name": "local"}))

    queue_pool.add_graph(graph_id, gname, "invalid configurations")

    cpool = load_client_pool({"name": "local"})
    with pytest.raises(ValueError, match=r"client pool not initialized"):
        queue_pool.get_client_pool()
    queue_pool.set_client_pool(cpool)
    with pytest.raises(ValueError, match=r"client pool already initialized"):
        queue_pool.set_client_pool(load_client_pool({"name": "local"}))

    with pytest.raises(ValueError, match=r"input node for .* not initialized"):
        queue_pool.get_input_node(graph_id)
    queue_pool.set_input_node(
        graph_id,
        load_node(
            graph,
            "constant_op",
            NodeId.create(gname, NName("input"))))
    with pytest.raises(
            ValueError, match=r"input node for .* already initialized"):
        queue_pool.set_input_node(
            graph_id,
            load_node(
                graph,
                "mat_square",
                NodeId.create(gname, NName("other"))))

    with pytest.raises(
            ValueError, match=r"output format for .* not initialized"):
        queue_pool.get_output_format(graph_id)
    queue_pool.set_output_format(
        graph_id, DataFormat({"out": DataInfo("int", [1, 2, 3])}))
    with pytest.raises(
            ValueError, match=r"output format for .* already initialized"):
        queue_pool.set_output_format(
            graph_id, DataFormat({"foo": DataInfo("float", [3, 2])}))

    node_id = NodeId.create(gname, NName("inner_1"))
    graph.add_node(
        queue_pool,
        graph_id,
        kind="constant_op",
        name=NName("inner_1"),
        node_id=node_id,
        args={},
        fixed_input_queue_id=None,
        vmap={})
    with pytest.raises(ValueError, match=r"node already added"):
        graph.add_node(
            queue_pool,
            graph_id,
            kind="mat_square",
            name=NName("valid"),
            node_id=node_id,
            args={},
            fixed_input_queue_id=None,
            vmap={})

    graph.add_node(
        queue_pool,
        graph_id,
        kind="constant_op",
        name=NName("inner_2"),
        node_id=None,
        args={},
        fixed_input_queue_id=QueueId.create(gname, NName("inner_2")),
        vmap={})
    with pytest.raises(ValueError, match=r"duplicate input queue id"):
        graph.add_node(
            queue_pool,
            graph_id,
            kind="constant_op",
            name=NName("inner_3"),
            node_id=None,
            args={},
            fixed_input_queue_id=QueueId.create(gname, NName("inner_2")),
            vmap={})

    with pytest.raises(ValueError, match=r"node strategy not set"):
        config.get_node_strategy()
    config.set_node_strategy(load_node_strategy({"name": "simple"}))
    config.set_node_strategy(load_node_strategy({"name": "simple"}))

    with pytest.raises(ValueError, match=r"queue strategy not set"):
        config.get_queue_strategy()
    config.set_queue_strategy(load_queue_strategy({"name": "simple"}))
    config.set_queue_strategy(load_queue_strategy({"name": "simple"}))


def test_ids() -> None:
    """Test invalid ids."""
    executor_id = ExecutorId.create()
    assert executor_id == ExecutorId.parse(executor_id.to_parseable())
    with pytest.raises(ValueError, match=r"invalid prefix for ExecutorId"):
        ExecutorId.parse("P5883f8c8a5bc4d56ae3202d1dc548118")
    with pytest.raises(ValueError, match=r"invalid ExecutorId"):
        ExecutorId.parse("Eec062b86-3d2c-4b89-a0ef-ff7fefd4487c")
    with pytest.raises(ValueError, match=r"invalid ExecutorId"):
        ExecutorId.parse("Eec062b86-3d2c-4b89-a0ef-ff7fefd4")

    ns = GNamespace("test")
    gname = QualifiedGraphName(ns, GName("test"))
    with pytest.raises(ValueError, match=r"name node:foo is not valid"):
        NodeId.create(gname, NName("node:foo"))
    with pytest.raises(ValueError, match=r"name node:foo is not valid"):
        QueueId.create(gname, NName("node:foo"))

    with pytest.raises(ValueError, match=r"queue id is output id"):
        QueueId.get_output_queue().ensure_queue()

    assert (
        QueueId.parse("Q5883f8c8a5bc4d56ae3202d1dc548118")
        != "Q5883f8c8a5bc4d56ae3202d1dc548118")
    assert (
        ExecutorId.parse("E5883f8c8a5bc4d56ae3202d1dc548118")
        != NodeId.parse("N5883f8c8a5bc4d56ae3202d1dc548118"))

    prefixes = set()

    def check_prefix(prefix: str) -> None:
        assert prefix not in prefixes
        prefixes.add(prefix)

    check_prefix(ExecutorId.prefix())
    check_prefix(GraphId.prefix())
    check_prefix(NodeId.prefix())
    check_prefix(QueueId.prefix())
    check_prefix(TaskId.prefix())
    check_prefix(DataId.prefix())

    data_id = LocalDataId.parse("D15")
    assert data_id == data_id  # pylint: disable=comparison-with-itself
    assert data_id == LocalDataId.parse(data_id.to_parseable())
    assert LocalDataId.parse("D0") == LocalDataId.parse("D0")
    assert LocalDataId.parse("D1") != LocalDataId.parse("D54321")
    assert LocalDataId.parse("D0") != "D0"
    with pytest.raises(ValueError, match=r"invalid prefix for LocalDataId"):
        LocalDataId.parse("E54321")
    with pytest.raises(ValueError, match=r"invalid id for LocalDataId"):
        LocalDataId.parse("Dtest")

    with pytest.raises(ValueError, match=r"value name :foo is not valid"):
        QualifiedName(None, ":foo")
    with pytest.raises(ValueError, match=r"is not a qualified name"):
        QualifiedName.parse("f o o")


@pytest.mark.parametrize("is_redis", [False, True])
def test_compute_task(is_redis: bool) -> None:
    """
    Test invalid tasks.

    Args:
        is_redis (bool): Whether we use redis.
    """
    config = load_test(is_redis=is_redis)
    cpool = config.get_client_pool()
    ns = GNamespace("test")
    task_id = cpool.create_task(ns, TaskValueContainer())
    task = ComputeTask(cpool, task_id, {})
    with pytest.raises(ValueError, match="no output data set"):
        task.get_data_out()
    with pytest.raises(ValueError, match="no output weight set"):
        task.get_weight_out()
    with pytest.raises(ValueError, match="no output byte size set"):
        task.get_byte_size_out()
    with pytest.raises(ValueError, match="no next queue id set"):
        task.get_next_queue_id()
    qid = QueueId.parse("Q5883f8c8a5bc4d56ae3202d1dc548118")
    task.set_result(DataContainer(), 1.0, 1, None, qid)
    with pytest.raises(ValueError, match="result already set"):
        task.set_result(DataContainer(), 2.0, 2, None, qid)
