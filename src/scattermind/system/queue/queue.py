from collections.abc import Iterable
from typing import TYPE_CHECKING

from scattermind.system.base import (
    ExecutorId,
    GraphId,
    Module,
    QueueId,
    TaskId,
)
from scattermind.system.client.client import ClientPool, ComputeTask
from scattermind.system.info import DataFormat
from scattermind.system.logger.context import ctx_fmt, get_ctx
from scattermind.system.logger.error import ErrorInfo, to_retry_event
from scattermind.system.logger.log import EventStream
from scattermind.system.names import GName, QualifiedName, ValueMap
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.queue.strategy.strategy import (
    NodeStrategy,
    QueueStrategy,
)
from scattermind.system.response import (
    TASK_STATUS_BUSY,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TASK_STATUS_WAIT,
    TaskStatus,
)


if TYPE_CHECKING:
    from scattermind.system.graph.node import Node


class Queue:
    def __init__(self, queue_pool: 'QueuePool', qid: QueueId) -> None:
        self._queue_pool = queue_pool
        self._qid = qid.ensure_queue()

    def get_id(self) -> QueueId:
        return self._qid

    def get_consumer_node(self) -> 'Node':
        return self._queue_pool.get_consumer_node(self._qid)

    def get_unclaimed_tasks(self) -> list[ComputeTask]:
        return self._queue_pool.get_unclaimed_compute_tasks(self._qid)

    def get_expected_new_task_weight(self) -> float:
        return self._queue_pool.get_expected_new_task_weight(self._qid)

    def get_queue_length(self) -> int:
        return self._queue_pool.get_queue_length(self._qid)

    def get_expected_byte_size(self) -> int:
        return self._queue_pool.get_expected_byte_size(self._qid)

    def get_incoming_byte_size(self) -> int:
        return self._queue_pool.get_incoming_byte_size(self._qid)

    def total_backpressure(self) -> float:
        res = 0.0
        for task in self.get_unclaimed_tasks():
            res += task.get_total_weight_in()
        return res

    def comparative_backpressure(self) -> float:
        res = 0.0
        for task in self.get_unclaimed_tasks():
            res += task.get_simple_weight_in()
        return res

    def total_expected_pressure(self) -> float:
        total_weight = self.get_expected_new_task_weight()
        return total_weight * self.get_expected_byte_size()

    def comparative_expected_pressure(self) -> float:
        return self.get_expected_new_task_weight()

    def claim_tasks(
            self,
            batch_size: int,
            executor_id: ExecutorId) -> list[ComputeTask]:
        return self._queue_pool.claim_compute_tasks(
            self._qid, batch_size, executor_id)

    def unclaim_tasks(self, executor_id: ExecutorId) -> list[TaskId]:
        return self._queue_pool.unclaim_tasks(self._qid, executor_id)

    def push_task_id(self, task_id: TaskId) -> None:
        return self._queue_pool.push_task_id(self._qid, task_id)


class QueuePool(Module):
    def __init__(self) -> None:
        super().__init__()
        self._client_pool: ClientPool | None = None
        self._entry_graph: GraphId | None = None
        self._graph_ids: dict[GName, GraphId] = {}
        self._graph_names: dict[GraphId, GName] = {}
        self._graph_descs: dict[GraphId, str] = {}
        self._input_nodes: dict[GraphId, 'Node'] = {}
        self._input_formats: dict[GraphId, DataFormat] = {}
        self._output_formats: dict[GraphId, DataFormat] = {}
        self._output_vmaps: dict[GraphId, ValueMap] = {}
        self._nodes: set['Node'] = set()
        self._input_queues: dict[QueueId, 'Node'] = {}
        self._node_strategy: NodeStrategy | None = None
        self._queue_strategy: QueueStrategy | None = None

    def set_client_pool(self, client_pool: ClientPool) -> None:
        if self._client_pool is not None:
            raise ValueError("client pool already initialized")
        self._client_pool = client_pool

    def get_client_pool(self) -> ClientPool:
        if self._client_pool is None:
            raise ValueError("client pool not initialized")
        return self._client_pool

    def set_entry_graph(self, graph_id: GraphId) -> None:
        if self._entry_graph is not None:
            raise ValueError("entry graph already initialized")
        self._entry_graph = graph_id

    def get_entry_graph(self) -> GraphId:
        if self._entry_graph is None:
            raise ValueError("entry graph not initialized")
        return self._entry_graph

    def add_graph(self, graph_id: GraphId, gname: GName, desc: str) -> None:
        if graph_id in self._graph_names:
            raise ValueError(f"duplicate graph {graph_id}")
        if gname in self._graph_ids:
            raise ValueError(f"duplicate graph name: {gname.get()}")
        self._graph_ids[gname] = graph_id
        self._graph_names[graph_id] = gname
        self._graph_descs[graph_id] = desc

    def get_graph_id(self, gname: GName) -> GraphId:
        return self._graph_ids[gname]

    def get_graph_name(self, graph_id: GraphId) -> GName:
        return self._graph_names[graph_id]

    def get_graphs(self) -> list[GraphId]:
        return list(self._graph_names.keys())

    def get_graph_description(self, graph_id: GraphId) -> str:
        return self._graph_descs[graph_id]

    def set_input_node(self, graph_id: GraphId, node: 'Node') -> None:
        if graph_id in self._input_nodes:
            raise ValueError(f"input node for {graph_id} already initialized")
        self._input_nodes[graph_id] = node

    def get_input_node(self, graph_id: GraphId) -> 'Node':
        res = self._input_nodes.get(graph_id)
        if res is None:
            raise ValueError(f"input node for {graph_id} not initialized")
        return res

    def set_input_format(
            self, graph_id: GraphId, input_format: DataFormat) -> None:
        if graph_id in self._input_formats:
            raise ValueError(
                f"input format for {graph_id} already initialized")
        self._input_formats[graph_id] = input_format

    def get_input_format(self, graph_id: GraphId) -> DataFormat:
        res = self._input_formats.get(graph_id)
        if res is None:
            raise ValueError(f"input format for {graph_id} not initialized")
        return res

    def set_output_format(
            self, graph_id: GraphId, output_format: DataFormat) -> None:
        if graph_id in self._output_formats:
            raise ValueError(
                f"output format for {graph_id} already initialized")
        self._output_formats[graph_id] = output_format

    def get_output_format(self, graph_id: GraphId) -> DataFormat:
        res = self._output_formats.get(graph_id)
        if res is None:
            raise ValueError(f"output format for {graph_id} not initialized")
        return res

    def set_output_value_map(self, graph_id: GraphId, vmap: ValueMap) -> None:
        if graph_id in self._output_vmaps:
            raise ValueError("output value map already initialized")
        self._output_vmaps[graph_id] = vmap

    def get_output_value_map(self, graph_id: GraphId) -> ValueMap:
        return self._output_vmaps[graph_id]

    def set_node_strategy(self, node_strategy: NodeStrategy) -> None:
        self._node_strategy = node_strategy

    def get_node_strategy(self) -> NodeStrategy:
        if self._node_strategy is None:
            raise ValueError("node strategy not set!")
        return self._node_strategy

    def set_queue_strategy(self, queue_strategy: QueueStrategy) -> None:
        self._queue_strategy = queue_strategy

    def get_queue_strategy(self) -> QueueStrategy:
        if self._queue_strategy is None:
            raise ValueError("queue strategy not set!")
        return self._queue_strategy

    def get_all_nodes(self) -> Iterable['Node']:
        yield from self._nodes

    def register_node(self, node: 'Node') -> None:
        qid = node.get_input_queue()
        if qid in self._input_queues:
            raise ValueError(
                f"duplicate input queue id: {qid} already registered "
                f"for {self._input_queues[qid]}")
        self._input_queues[qid] = node
        self._nodes.add(node)

    def get_queue(self, qid: QueueId) -> Queue:
        return Queue(self, qid)

    def get_consumer_node(self, qid: QueueId) -> 'Node':
        return self._input_queues[qid]

    def get_all_queues(self) -> Iterable[QueueId]:
        yield from self._input_queues

    def pick_node(
            self,
            logger: EventStream,
            current_node: 'Node | None') -> tuple['Node', bool]:
        strategy = self.get_node_strategy()
        entry_graph_id = self.get_entry_graph()
        candidate_node = self.get_input_node(entry_graph_id)
        candidate_score = 0.0
        for node in self.get_all_nodes():
            cur_graph_id = node.get_graph()
            cur_graph_name = self.get_graph_name(cur_graph_id)
            qid = node.get_input_queue()
            queue = self.get_queue(qid)
            length = queue.get_queue_length()
            pressure = queue.total_backpressure()
            expected_pressure = queue.total_expected_pressure()
            score = strategy.other_score(
                queue_length=length,
                pressure=pressure,
                expected_pressure=expected_pressure,
                cost_to_load=node.get_load_cost())
            logger.log_event(
                "measure.queue.input",
                {
                    "name": "queue_input",
                    "length": length,
                    "pressure": pressure,
                    "expected_pressure": expected_pressure,
                    "score": score,
                },
                adjust_ctx={
                    "executor": None,
                    "task": None,
                    "graph": cur_graph_id,
                    "graph_name": cur_graph_name,
                    "node": node.get_id(),
                    "node_name": node.get_name(),
                })
            if score > candidate_score:
                candidate_score = score
                candidate_node = node
        if current_node is None:
            return (candidate_node, True)
        qid = current_node.get_input_queue()
        own_queue = self.get_queue(qid)
        own_length = own_queue.get_queue_length()
        own_pressure = own_queue.total_backpressure()
        own_expected_pressure = own_queue.total_expected_pressure()
        own_score = strategy.own_score(
            queue_length=own_length,
            pressure=own_pressure,
            expected_pressure=own_expected_pressure,
            cost_to_load=current_node.get_load_cost())
        if strategy.want_to_switch(own_score, candidate_score):
            return (candidate_node, candidate_node != current_node)
        return (current_node, False)

    def enqueue_task(
            self,
            store: DataStore,
            original_input: TaskValueContainer) -> TaskId:
        cpool = self.get_client_pool()
        task_id = cpool.create_task(original_input)
        self._enqueue_task_id(cpool, store, task_id)
        return task_id

    def maybe_requeue_task_id(
            self,
            logger: EventStream,
            store: DataStore,
            task_id: TaskId,
            error_info: ErrorInfo) -> None:
        cpool = self.get_client_pool()
        cpool.set_error(task_id, error_info)
        new_retries = cpool.inc_retries(task_id)
        adj_ctx, retry_event = to_retry_event(error_info, new_retries)
        logger.log_event("error.task", retry_event, adjust_ctx=adj_ctx)
        if new_retries >= ClientPool.get_max_retries():
            cpool.set_duration(task_id)
            cpool.set_bulk_status([task_id], TASK_STATUS_ERROR)
        else:
            if cpool.get_status(task_id) == TASK_STATUS_WAIT:
                mqid = self.maybe_get_queue(task_id)
                if mqid is not None:
                    raise AssertionError(
                        f"{ctx_fmt()} {task_id} already waiting at {mqid}")
            cpool.clear_progress(task_id)
            self._enqueue_task_id(cpool, store, task_id)

    def handle_task_result(
            self,
            logger: EventStream,
            store: DataStore,
            task: ComputeTask) -> None:
        cpool = self.get_client_pool()
        task_id = task.get_task_id()
        cpool.commit_task(
            task_id,
            task.get_data_out(),
            weight=task.get_weight_out(),
            byte_size=task.get_byte_size_out(),
            push_frame=task.get_push_frame())
        qid = task.get_next_queue_id()
        is_final = False
        while not is_final and qid.is_output_id():
            next_frame, frame_data = cpool.pop_frame(task_id)
            if next_frame is None:
                m_nname = None
                graph_id = self.get_entry_graph()
                is_final = True
            else:
                m_nname, graph_id, qid = next_frame
            vmap = self.get_output_value_map(graph_id)
            ret_data = {
                QualifiedName(m_nname, vname): frame_data[frame_qual]
                for vname, frame_qual in vmap.items()
            }
            if is_final:
                final_vmap = {
                    qual.get_value_name(): qual
                    for qual, _ in ret_data.items()
                }
                output_format = self.get_output_format(graph_id)
                tvc_out = TaskValueContainer.extract_data(
                    store, output_format, ret_data, final_vmap)
                if tvc_out is not None:
                    cpool.set_final_output(task_id, tvc_out)
                    cpool.set_duration(task_id)
                    cpool.set_bulk_status([task_id], TASK_STATUS_READY)
                else:
                    self.maybe_requeue_task_id(
                        logger,
                        store,
                        task_id,
                        {
                            "ctx": get_ctx(),
                            "message": "result got purged from memory",
                            "code": "memory_purge",
                            "traceback": [],
                        })
            else:
                cpool.commit_task(
                    task_id,
                    ret_data,
                    weight=task.get_weight_out(),
                    byte_size=task.get_byte_size_out(),
                    push_frame=None)
        if not is_final:
            out_queue = self.get_queue(qid)
            out_queue.push_task_id(task_id)

    def get_task_status(self, task_id: TaskId) -> TaskStatus:
        return self.get_client_pool().get_status(task_id)

    def _enqueue_task_id(
            self,
            cpool: ClientPool,
            store: DataStore,
            task_id: TaskId) -> None:
        entry_graph_id = self.get_entry_graph()
        input_format = self.get_input_format(entry_graph_id)
        cpool.init_data(store, task_id, input_format)
        node = self.get_input_node(entry_graph_id)
        qid = node.get_input_queue()
        cpool.set_bulk_status([task_id], TASK_STATUS_WAIT)
        self.push_task_id(qid, task_id)

    def get_compute_task(
            self,
            cpool: ClientPool,
            qid: QueueId,
            task_id: TaskId) -> ComputeTask:
        node = self.get_consumer_node(qid)
        return ComputeTask(cpool, task_id, node.get_value_map())

    def claim_compute_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[ComputeTask]:
        cpool = self.get_client_pool()
        task_ids = cpool.set_bulk_status(
            self.claim_tasks(qid, batch_size, executor_id), TASK_STATUS_BUSY)
        return [
            self.get_compute_task(cpool, qid, task_id)
            for task_id in task_ids
        ]

    def get_unclaimed_compute_tasks(self, qid: QueueId) -> list[ComputeTask]:
        cpool = self.get_client_pool()
        return [
            self.get_compute_task(cpool, qid, task_id)
            for task_id in self.get_unclaimed_tasks(qid)
        ]

    def sort_tasks(self, task_ids: list[TaskId]) -> None:
        strategy = self.get_queue_strategy()
        strategy.sort_queue(self.get_client_pool(), task_ids)

    def push_task_id(self, qid: QueueId, task_id: TaskId) -> None:
        raise NotImplementedError()

    def get_unclaimed_tasks(self, qid: QueueId) -> list[TaskId]:
        raise NotImplementedError()

    def claim_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[TaskId]:
        raise NotImplementedError()

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        raise NotImplementedError()

    def expect_task_weight(
            self,
            weight: float,
            byte_size: int,
            qid: QueueId,
            executor_id: ExecutorId) -> None:
        raise NotImplementedError()

    def clear_expected_task_weight(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        raise NotImplementedError()

    def get_expected_new_task_weight(self, qid: QueueId) -> float:
        raise NotImplementedError()

    def get_queue_length(self, qid: QueueId) -> int:
        raise NotImplementedError()

    def get_expected_byte_size(self, qid: QueueId) -> int:
        raise NotImplementedError()

    def get_incoming_byte_size(self, qid: QueueId) -> int:
        raise NotImplementedError()

    def maybe_get_queue(self, task_id: TaskId) -> QueueId | None:
        raise NotImplementedError()
