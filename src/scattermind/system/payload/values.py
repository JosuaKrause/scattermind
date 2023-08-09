from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from scattermind.system.base import DataId, QueueId
from scattermind.system.client.client import ComputeTask
from scattermind.system.helper import DictHelper
from scattermind.system.info import DataFormat, DataInfo
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.names import (
    GName,
    NName,
    QName,
    QualifiedName,
    ValueMap,
)
from scattermind.system.torch_util import (
    extract_shapes,
    mask_from_shapes,
    pad_list,
)


if TYPE_CHECKING:
    from scattermind.system.graph.node import Node
    from scattermind.system.payload.data import DataStore
    from scattermind.system.queue.queue import QueuePool


DataContainer = dict[QualifiedName, DataId]
"""A data container contains all data stored in a stack frame."""


class NoTasksToCompute(Exception):
    pass


class TaskValueContainer(DictHelper[torch.Tensor]):
    def place_data(
            self,
            nname: NName | None,
            store: 'DataStore',
            data_format: DataFormat,
            data_out: DataContainer) -> None:
        for vname, data_info in data_format.items():
            value = data_info.check_tensor(self[vname])
            data_id = store.store_tensor(value)
            data_out[QualifiedName(nname, vname)] = data_id

    @staticmethod
    def extract_data(
            store: 'DataStore',
            data_format: DataFormat,
            data_in: DataContainer,
            vmap: ValueMap) -> 'TaskValueContainer | None':

        def get_content(vname: str, data_id: DataId) -> torch.Tensor | None:
            data_info = data_format[vname]
            res = store.get_tensor(data_id, data_info)
            if res is None:
                return None
            return res

        if set(vmap) != set(data_format.keys()):
            raise ValueError("value map does not match data format")
        res = TaskValueContainer()
        for vname, qual in vmap.items():
            data_id = data_in[qual]
            content = get_content(vname, data_id)
            if content is None:
                return None
            res[vname] = content
        return res

    def byte_size(self, data_format: DataFormat) -> int:
        res = 0
        for name, data_info in data_format.items():
            res += data_info.byte_size(list(self[name].shape))
        return res


class ComputeValues:
    def __init__(
            self,
            data_info: DataInfo,
            *,
            values_list: list[torch.Tensor] | None = None,
            values_uniform: torch.Tensor | None = None,
            values_padded: tuple[torch.Tensor, list[list[int]]] | None = None,
            ) -> None:
        if sum([
                values_list is not None,
                values_uniform is not None,
                values_padded is not None]) != 1:
            raise ValueError("one value must be non-None")
        self._values_list = values_list
        self._values_uniform = values_uniform
        self._values_padded = values_padded
        self._data_info = data_info
        self._validate()

    def _validate(self) -> None:
        for value in self.iter_values():
            self._data_info.check_tensor(value)

    @staticmethod
    def create_uniform(
            data_info: DataInfo, values: torch.Tensor) -> 'ComputeValues':
        return ComputeValues(data_info, values_uniform=values)

    @staticmethod
    def create_padded(
            data_info: DataInfo,
            values: torch.Tensor,
            shapes: list[list[int]]) -> 'ComputeValues':
        return ComputeValues(data_info, values_padded=(values, shapes))

    @classmethod
    def create_masked(
            cls,
            data_info: DataInfo,
            values: torch.Tensor,
            mask: torch.Tensor) -> 'ComputeValues':
        return cls.create_padded(data_info, values, extract_shapes(mask))

    def row_count(self) -> int:
        if self._values_list is not None:
            return len(self._values_list)
        if self._values_uniform is not None:
            return list(self._values_uniform.shape)[0]
        if self._values_padded is not None:
            _, shapes = self._values_padded
            return len(shapes)
        raise ValueError("one value must be non-None")

    def iter_values(self) -> Iterable[torch.Tensor]:
        if self._values_list is not None:
            yield from self._values_list
        elif self._values_uniform is not None:
            values = self._values_uniform
            yield from (
                values[ix, :]
                for ix in range(values.shape[0])
            )
        elif self._values_padded is not None:
            values, shapes = self._values_padded
            yield from (
                values[[ix] + [slice(dim) for dim in shape]]
                for ix, shape in enumerate(shapes)
            )
        else:
            raise ValueError("one value must be non-None")

    def is_converted(self) -> bool:
        return self._values_list is None

    def is_uniform(self) -> bool:
        return self._data_info.is_uniform()

    def convert(self) -> None:
        if self._values_list is None:
            return
        values = self._values_list
        if self.is_uniform():
            self._values_uniform = torch.vstack([
                torch.unsqueeze(val, 0)
                for val in values
            ])
        else:
            shapes = [
                list(value.shape)
                for value in values
            ]
            padded = pad_list(values, self._data_info.max_shape(shapes))
            self._values_padded = padded, shapes
        self._values_list = None

    def get_uniform(self) -> torch.Tensor:
        if self._values_list is not None:
            self.convert()
        if self._values_uniform is not None:
            return self._values_uniform
        raise ValueError("cannot create uniform tensor for varying shapes")

    def get_masked(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._values_list is not None:
            self.convert()
        if self._values_padded is not None:
            values, shapes = self._values_padded
            return values, mask_from_shapes(shapes, list(values.shape)[1:])
        raise ValueError("cannot create mask tensor for uniform shapes")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}["
            f"info={self._data_info},"
            f"rows={self.row_count()}]")

    def __repr__(self) -> str:
        return self.__str__()


class ComputeValueContainer:
    def __init__(
            self,
            data_format: DataFormat,
            store: 'DataStore',
            tasks: list[ComputeTask]) -> None:
        self._data_format = data_format
        self._store = store
        self._tasks = tasks
        self._rejected_tasks: list[ComputeTask] | None = None
        self._data: dict[str, ComputeValues] | None = None

    def _compute_data(self) -> dict[str, ComputeValues]:
        if self._data is not None:
            return self._data
        assert self._rejected_tasks is None
        tasks = self._tasks
        store = self._store
        data_format = self._data_format
        rejected_tasks: list[ComputeTask] = []
        accepted_tasks: list[ComputeTask] = []
        data: dict[str, list[torch.Tensor]] | None = None
        for task in tasks:
            data_in = task.get_data_in()
            rejected = False
            cur = {}
            for name, data_id in data_in.items():
                val = store.get_tensor(data_id, data_format[name])
                if val is None:
                    rejected = True
                    break
                cur[name] = val
            if rejected:
                rejected_tasks.append(task)
                continue
            accepted_tasks.append(task)
            if data is None:
                data = {
                    name: []
                    for name in data_in.keys()
                }
            for name, val in cur.items():
                data[name].append(val)
        self._tasks = accepted_tasks
        self._rejected_tasks = rejected_tasks
        if data is None:
            self._data = {}
            raise NoTasksToCompute()
        self._data = {
            name: ComputeValues(data_format[name], values_list=vals)
            for name, vals in data.items()
        }
        return self._data

    def get_current_rejected_tasks(self) -> list[ComputeTask]:
        return [] if self._rejected_tasks is None else self._rejected_tasks

    def get_current_tasks(self) -> list[ComputeTask]:
        return self._tasks

    def get_data(self, name: str) -> ComputeValues:
        data = self._compute_data()
        return data[name]

    def get_value_names(self) -> list[str]:
        return list(self._data_format.keys())

    def iter_values(self) -> Iterable[
            tuple[ComputeTask, dict[str, torch.Tensor]]]:
        data = self._compute_data()
        res: list[dict[str, torch.Tensor]] = [{} for _ in self._tasks]
        for key in self._data_format.keys():
            for ix, value in enumerate(data[key].iter_values()):
                res[ix][key] = value
        for ix, task in enumerate(self._tasks):
            yield task, res[ix]

    def count(self) -> int:
        return len(self._tasks)


class ComputeState:
    def __init__(
            self,
            queue_pool: 'QueuePool',
            store: 'DataStore',
            node: 'Node',
            tasks: list[ComputeTask]) -> None:
        self._queue_pool = queue_pool
        self._store = store
        self._node = node
        self._data_in = ComputeValueContainer(
            node.get_input_data_format(), store, tasks)
        self._tasks_out: list[ComputeTask] = []

    def get_inputs_tasks(self) -> Iterable[ComputeTask]:
        yield from self._data_in.get_current_tasks()

    def get_current_rejected_tasks(self) -> Iterable[ComputeTask]:
        yield from self._data_in.get_current_rejected_tasks()

    def get_values(self) -> ComputeValueContainer:
        return self._data_in

    def create_single(self, value: torch.Tensor) -> 'LazyValues':
        return self.create_uniform(torch.unsqueeze(value, 0))

    def create_uniform(self, value: torch.Tensor) -> 'LazyValues':
        return LazyValues(values_uniform=value)

    def create_padded(
            self,
            values: torch.Tensor,
            shapes: list[list[int]]) -> 'LazyValues':
        return LazyValues(values_padded=(values, shapes))

    def create_masked(
            self,
            value: torch.Tensor,
            mask: torch.Tensor) -> 'LazyValues':
        return LazyValues(values_masked=(value, mask))

    def create(self, values: ComputeValues) -> 'LazyValues':
        if values.is_uniform():
            return self.create_uniform(values.get_uniform())
        return self.create_masked(*values.get_masked())

    def get_graph_input_queue(self, gname: GName) -> QueueId:
        queue_pool = self._queue_pool
        graph_id = queue_pool.get_graph_id(gname)
        node = queue_pool.get_input_node(graph_id)
        return node.get_input_queue()

    def push_call(
            self,
            qname: str,
            tasks: list[ComputeTask],
            args: dict[str, 'LazyValues'],
            gname: GName) -> None:
        qname_obj = QName(qname)
        queue_pool = self._queue_pool
        store = self._store
        node = self._node
        caller_name = node.get_name()
        graph_id = queue_pool.get_graph_id(gname)
        next_qid = self.get_graph_input_queue(gname)
        data_format = queue_pool.get_input_format(graph_id)
        data = {
            key: args[key].to_compute_values(data_info)
            for key, data_info in data_format.items()
        }
        print(f"{ctx_fmt()} push_call data={data} data_format={data_format}")
        data_ids: list[DataContainer] = [{} for _ in tasks]
        byte_sizes: list[int] = [0 for _ in tasks]
        for key, cvalues in data.items():
            qual = QualifiedName(None, key)
            for ix, value in enumerate(cvalues.iter_values()):
                data_ids[ix][qual] = store.store_tensor(value)
                byte_sizes[ix] += data_format[key].byte_size(list(value.shape))
        add_weight = node.get_weight()
        return_qid = node.get_output_queue(qname_obj)
        for task, dids, byte_size in zip(tasks, data_ids, byte_sizes):
            task.set_result(
                dids,
                add_weight,
                byte_size,
                (caller_name, graph_id, return_qid),
                next_qid)
            self._tasks_out.append(task)

    def push_results(
            self,
            qname: str,
            tasks: list[ComputeTask],
            values: dict[str, 'LazyValues']) -> None:
        qname_obj = QName(qname)
        store = self._store
        node = self._node
        data_format = self._node.get_output_data_format(qname_obj)
        data = {
            key: values[key].to_compute_values(data_info)
            for key, data_info in data_format.items()
        }
        data_ids: list[DataContainer] = [{} for _ in tasks]
        byte_sizes: list[int] = [0 for _ in tasks]
        for key, cvalues in data.items():
            qual = QualifiedName(node.get_name(), key)
            for ix, value in enumerate(cvalues.iter_values()):
                data_ids[ix][qual] = store.store_tensor(value)
                byte_sizes[ix] += data_format[key].byte_size(list(value.shape))
        add_weight = node.get_weight()
        next_qid = node.get_output_queue(qname_obj)
        for task, dids, byte_size in zip(tasks, data_ids, byte_sizes):
            task.set_result(dids, add_weight, byte_size, None, next_qid)
            self._tasks_out.append(task)

    def verify_results(self) -> None:
        tasks_in = {
            task.get_task_id()
            for task in self._data_in.get_current_tasks()
        }
        tasks_out = {
            task.get_task_id()
            for task in self._tasks_out
        }
        if tasks_in != tasks_out:
            raise ValueError(
                "unbalanced input to output tasks "
                f"{tasks_in.symmetric_difference(tasks_out)}")

    def results(self) -> Iterable[ComputeTask]:
        self.verify_results()
        yield from self._tasks_out

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}["
            f"node={self._node},"
            f"tasks={self._data_in.count()}]")

    def __repr__(self) -> str:
        return self.__str__()


class LazyValues:  # pylint: disable=too-few-public-methods
    def __init__(
            self,
            *,
            values_uniform: torch.Tensor | None = None,
            values_padded: tuple[torch.Tensor, list[list[int]]] | None = None,
            values_masked: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> None:
        if sum([
                values_uniform is not None,
                values_padded is not None,
                values_masked is not None]) != 1:
            raise ValueError("only one argument allowed")
        self._values_uniform = values_uniform
        self._values_padded = values_padded
        self._values_masked = values_masked

    def to_compute_values(self, data_info: DataInfo) -> ComputeValues:
        if self._values_uniform is not None:
            return ComputeValues.create_uniform(
                data_info, self._values_uniform)
        if self._values_padded is not None:
            values, shapes = self._values_padded
            return ComputeValues.create_padded(data_info, values, shapes)
        if self._values_masked is not None:
            values, mask = self._values_masked
            return ComputeValues.create_masked(data_info, values, mask)
        raise ValueError("cannot happen")

    def shape(self) -> list[int | None]:
        if self._values_uniform is not None:
            return list(self._values_uniform.shape)
        if self._values_padded is not None:
            _, shapes = self._values_padded
            if not shapes:
                return [0]
            count: list[int | None] = [len(shapes)]
            max_shape: list[int | None] = list(shapes[0])
            for shape in shapes[1:]:
                for ix, dim in enumerate(shape):
                    if max_shape[ix] != dim:
                        max_shape[ix] = None
            return count + max_shape
        if self._values_masked is not None:
            _, mask = self._values_masked
            return list(mask.shape)
        raise ValueError("cannot happen")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.shape()}]"

    def __repr__(self) -> str:
        return self.__str__()
