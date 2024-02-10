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
"""Provides classes to handle payload data efficiently."""
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from scattermind.system.base import DataId, QueueId
from scattermind.system.client.client import ComputeTask
from scattermind.system.helper import DictHelper
from scattermind.system.info import DataFormat, DataInfo
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.names import (
    NName,
    QName,
    QualifiedGraphName,
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
    """
    Raised when there are no tasks left to compute. Note, this is different
    from an empty queue. When loading payload data for a task it can happen
    that some part of the payload data has been purged from memory. This
    happens at the discretion of the payload data store. If some data needed
    by a task for the current execution step has been purged the task cannot
    continue execution, cannot computed in full, and has to be requeued from
    the beginning. If all tasks claimed from a queue cannot be computed because
    of this situation, there is nothing left to compute for the current node
    and this exception is raised.
    """


class TaskValueContainer(DictHelper[torch.Tensor]):
    """Contains the data for one task. Keys are the field names and values
    are the tensors."""
    def place_data(
            self,
            nname: NName | None,
            store: 'DataStore',
            data_format: DataFormat,
            data_out: DataContainer) -> None:
        """
        Place the data into the payload data storage. The resulting data ids
        are stored in the data container.

        Args:
            nname (NName | None): The name of the node.
            store (DataStore): The payload data store.
            data_format (DataFormat): The expected data format.
            data_out (DataContainer): The data container.
        """
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
        """
        Reads data from a data container using a value map. If any data is
        missing from the payload data store (e.g. if some data was purged)
        None is returned.

        Args:
            store (DataStore): The payload data storage.
            data_format (DataFormat): The expected data format.
            data_in (DataContainer): The data to read.
            vmap (ValueMap): The data placement. The value map defines which
                keys will be available in the task value container and where
                in the data container the data comes from.

        Returns:
            TaskValueContainer | None: The task value container if all data
                is available. None otherwise.
        """

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
        """
        Compute the byte size of the task value container.

        Args:
            data_format (DataFormat): The expected data format.

        Returns:
            int: The byte size.
        """
        res = 0
        for name, data_info in data_format.items():
            res += data_info.byte_size(list(self[name].shape))
        return res


class ComputeValues:
    """Represents a batch tensor and offers various ways of accessing the
    tensor of individual tasks or the whole batch. A batch tensor combines
    multiple rows (tasks). Compute values can be created in various ways and
    read in various ways. Reading and writing methods do not have to match
    and the data will be converted between them as needed."""
    def __init__(
            self,
            data_info: DataInfo,
            *,
            values_list: list[torch.Tensor] | None = None,
            values_uniform: torch.Tensor | None = None,
            values_padded: tuple[torch.Tensor, list[list[int]]] | None = None,
            ) -> None:
        """
        Creates compute values. Exactly one input argument needs to be passed.
        Use the static factory methods instead of the constructor. The
        constructor is most commonly only used for tensor lists.

        Args:
            data_info (DataInfo): Information about the expected data.
            values_list (list[torch.Tensor] | None, optional): Creates from
                a list of individual tensors. No additional dimension
                representing rows (tasks) is needed. Each tensor is exactly
                the tensor for its corresponding task.
            values_uniform (torch.Tensor | None, optional): Creates from a
                uniform batch tensor. The first dimension represents the rows
                (tasks). All rows need to have the same shape. Defaults to
                None.
            values_padded (tuple[torch.Tensor, list[list[int]]] | None,
                optional): Creates from a padded tensor. A padded tensor is
                a tuple of the padded tensor (of maximum shape) and a list of
                each individual shape. The first dimension of the tensor
                represents the rows (tasks). Defaults to None.

        Raises:
            ValueError: If not exactly one argument is set.
        """
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
        """
        Create compute values from uniform data.

        Args:
            data_info (DataInfo): Information about the expected data.
            values (torch.Tensor): The batch tensor containing all rows (tasks)
                in a unified shape. All rows have the same shape. The shape of
                the tensor has an extra dimension to separate rows.

        Returns:
            ComputeValues: The compute values.
        """
        return ComputeValues(data_info, values_uniform=values)

    @staticmethod
    def create_padded(
            data_info: DataInfo,
            values: torch.Tensor,
            shapes: list[list[int]]) -> 'ComputeValues':
        """
        Create compute values from padded data.

        Args:
            data_info (DataInfo): Information about the expected data.
            values (torch.Tensor): The padded tensor. The shape is the maximum
                shape of all rows (tasks) and includes an extra first dimension
                to separate rows.
            shapes (list[list[int]]): A list containing each individual shape
                of the tasks.

        Returns:
            ComputeValues: The compute values.
        """
        return ComputeValues(data_info, values_padded=(values, shapes))

    @classmethod
    def create_masked(
            cls,
            data_info: DataInfo,
            values: torch.Tensor,
            mask: torch.Tensor) -> 'ComputeValues':
        """
        Create compute values from masked data.

        Args:
            data_info (DataInfo): Information about the expected data.
            values (torch.Tensor): The padded tensor. The shape is the maximum
                shape of all rows (tasks) and includes an extra first dimension
                to separate rows.
            mask (torch.Tensor): The mask tensor. The mask tensor has the same
                shape as the padded tensor and indicates which elements of the
                value tensor are valid.

        Returns:
            ComputeValues: The compute values.
        """
        return cls.create_padded(data_info, values, extract_shapes(mask))

    def row_count(self) -> int:
        """
        Retrieves the number of rows.

        Returns:
            int: The number of rows (tasks).
        """
        if self._values_list is not None:
            return len(self._values_list)
        if self._values_uniform is not None:
            return int(list(self._values_uniform.shape)[0])
        if self._values_padded is not None:
            _, shapes = self._values_padded
            return len(shapes)
        raise ValueError("one value must be non-None")

    def iter_values(self) -> Iterable[torch.Tensor]:
        """
        Iterates through all rows one task at a time.

        Yields:
            torch.Tensor: The data of the row (task).
        """
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
        """
        Whether the tensor data has been converted to a single unified tensor
        (uniform, padded, or masked) instead of a list of tensors.

        Returns:
            bool: Whether the tensor data has been converted to a single
                tensor.
        """
        return self._values_list is None

    def is_uniform(self) -> bool:
        """
        Whether the data is uniform.

        Returns:
            bool: True, if the tensors of all tasks have the same shape.
        """
        return self._data_info.is_uniform()

    def convert(self) -> None:
        """
        Convert data to a unified tensor representation (uniform or padded).
        """
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
        """
        Get the data as uniform tensor.

        Raises:
            ValueError: If the data cannot be converted to a uniform
                representation. This is the case if shapes can differ between
                rows (tasks).

        Returns:
            torch.Tensor: The uniform tensor.
        """
        if self._values_list is not None:
            self.convert()
        if self._values_uniform is not None:
            return self._values_uniform
        raise ValueError("cannot create uniform tensor for varying shapes")

    def get_masked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the data as masked tensor.

        Raises:
            ValueError: If the data cannot be converted to a masked
                representation.This is the case if shapes cannot differ between
                rows (tasks).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The padded and masked tensor.
        """
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
    """
    Provides data for a collection of tasks in a streamlined manner.
    """
    def __init__(
            self,
            data_format: DataFormat,
            store: 'DataStore',
            tasks: list[ComputeTask]) -> None:
        """
        Create a compute value container providing access to the relevant data
        of a collection of tasks.

        Args:
            data_format (DataFormat): The expected data format.
            store (DataStore): The payload data store.
            tasks (list[ComputeTask]): The list of tasks.
        """
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
        data_id_type = store.data_id_type()
        data_format = self._data_format
        rejected_tasks: list[ComputeTask] = []
        accepted_tasks: list[ComputeTask] = []
        data: dict[str, list[torch.Tensor]] | None = None
        for task in tasks:
            data_in = task.get_data_in(data_id_type)
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
        """
        Retrieve the list of rejected tasks. A task gets rejected if some of
        its required data is missing (e.g., by being purged from the payload
        data storage).

        Returns:
            list[ComputeTask]: The list of rejected tasks. Note, that the list
                is empty before attempting to access the data of tasks.
        """
        return [] if self._rejected_tasks is None else self._rejected_tasks

    def get_current_tasks(self) -> list[ComputeTask]:
        """
        Retrieve the list of current / accepted tasks in order. A task is
        accepted if it was possible to fully load the relevant data. Before
        any attempt to load the data of a task has happened the task is current
        and will be returned as well.

        Returns:
            list[ComputeTask]: The list of accepted tasks. Note, that the list
                is full before attempting to access the data of the tasks. If
                you want this call to retrieve the actual list of accepted
                tasks call it after accessing the data.
        """
        return self._tasks

    def get_data(self, name: str) -> ComputeValues:
        """
        Get the data of all tasks for the given field.

        Args:
            name (str): The given value field.

        Returns:
            ComputeValues: The data of all tasks.
        """
        data = self._compute_data()
        return data[name]

    def get_value_names(self) -> list[str]:
        """
        Retrieves the list of all value fields.

        Returns:
            list[str]: The value fields.
        """
        return list(self._data_format.keys())

    def iter_values(self) -> Iterable[
            tuple[ComputeTask, dict[str, torch.Tensor]]]:
        """
        Iterate through all tasks and retrieve the data for each task
        individually.

        Yields:
            tuple[ComputeTask, dict[str, torch.Tensor]]: A tuple of the
                task and a dictionary mapping all value fields to the
                payload data.
        """
        data = self._compute_data()
        res: list[dict[str, torch.Tensor]] = [{} for _ in self._tasks]
        for key in self._data_format.keys():
            for ix, value in enumerate(data[key].iter_values()):
                res[ix][key] = value
        for ix, task in enumerate(self._tasks):
            yield task, res[ix]

    def count(self) -> int:
        """
        How many tasks are current / accepted.

        Returns:
            int: If called before retrieving the data this returns the number
                of tasks in the chunk. If called afterwards it returns the
                number of accepted tasks.
        """
        return len(self._tasks)


class ComputeState:
    """
    The full state of a collection of tasks needed to execute a given node.
    The state provides different views on the data of the tasks and
    functionality to set the results of the node.
    """
    def __init__(
            self,
            queue_pool: 'QueuePool',
            store: 'DataStore',
            node: 'Node',
            tasks: list[ComputeTask]) -> None:
        """
        Creates a compute state.

        Args:
            queue_pool (QueuePool): The queue pool.
            store (DataStore): The payload data store.
            node (Node): The executing node.
            tasks (list[ComputeTask]): The list of tasks.
        """
        self._queue_pool = queue_pool
        self._store = store
        self._node = node
        self._data_in = ComputeValueContainer(
            node.get_input_data_format(), store, tasks)
        self._tasks_out: list[ComputeTask] = []

    def get_inputs_tasks(self) -> Iterable[ComputeTask]:
        """
        Retrieve current / accepted tasks.

        Yields:
            ComputeTask: If the function is called after reading the data this
                function returns all accepted tasks. If the data has not been
                read yet this function returns all tasks.
        """
        yield from self._data_in.get_current_tasks()

    def get_current_rejected_tasks(self) -> Iterable[ComputeTask]:
        """
        Retrieve rejected tasks.

        Yields:
            ComputeTask: If the function is called after reading the data this
                function returns all rejected tasks. If the data has not been
                read yet this function returns no tasks.
        """
        yield from self._data_in.get_current_rejected_tasks()

    def get_values(self) -> ComputeValueContainer:
        """
        Get access to the data for the tasks.

        Returns:
            ComputeValueContainer: The access to the data for the tasks.
        """
        return self._data_in

    def create_single(self, value: torch.Tensor) -> 'LazyValues':
        """
        Create a lazy value for a single row (task).

        Args:
            value (torch.Tensor): The tensor for one task.

        Returns:
            LazyValues: The lazy value.
        """
        return self.create_uniform(torch.unsqueeze(value, 0))

    def create_uniform(self, value: torch.Tensor) -> 'LazyValues':
        """
        Create a lazy value for uniform rows (tasks).

        Args:
            value (torch.Tensor): The tensor for uniform tasks.

        Returns:
            LazyValues: The lazy value.
        """
        return LazyValues(values_uniform=value)

    def create_padded(
            self,
            values: torch.Tensor,
            shapes: list[list[int]]) -> 'LazyValues':
        """
        Create a lazy value for padded rows (tasks).

        Args:
            values (torch.Tensor): The tensor for padded tasks.
            shapes (list[list[int]]): The corresponding shapes.

        Returns:
            LazyValues: The lazy value.
        """
        return LazyValues(values_padded=(values, shapes))

    def create_masked(
            self,
            value: torch.Tensor,
            mask: torch.Tensor) -> 'LazyValues':
        """
        Create a lazy value for masked rows (tasks).

        Args:
            value (torch.Tensor): The tensor for padded tasks.
            mask (torch.Tensor): The corresponding mask.

        Returns:
            LazyValues: The lazy value.
        """
        return LazyValues(values_masked=(value, mask))

    def create(self, values: ComputeValues) -> 'LazyValues':
        """
        Create a lazy value choosing the format depending on the type of data.

        Args:
            values (ComputeValues): The values.

        Returns:
            LazyValues: The lazy values.
        """
        if values.is_uniform():
            return self.create_uniform(values.get_uniform())
        return self.create_masked(*values.get_masked())

    def get_graph_input_queue(self, gname: QualifiedGraphName) -> QueueId:
        """
        Retrieve the input queue of the graph.

        Args:
            gname (QualifiedGraphName): The qualified name of the graph.

        Returns:
            QueueId: The queue id.
        """
        queue_pool = self._queue_pool
        graph_id = queue_pool.get_graph_id(gname)
        node = queue_pool.get_input_node(graph_id)
        return node.get_input_queue()

    def push_call(
            self,
            qname: str,
            tasks: list[ComputeTask],
            args: dict[str, 'LazyValues'],
            gname: QualifiedGraphName) -> None:
        """
        Push a call to a subgraph.

        Args:
            qname (str): Where execution returns after the subgraph
                computation has finished. The name of the output of the node.
            tasks (list[ComputeTask]): The tasks to push.
            args (dict[str, LazyValues]): The input for the entry node of the
                subgraph.
            gname (QualifiedGraphName): The subgraph to call.
        """
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
        """
        Push the result of the current node.

        Args:
            qname (str): The name of the output of the node.
            tasks (list[ComputeTask]): The tasks to push.
            values (dict[str, LazyValues]): The result values.
        """
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
        """
        Verify the results that have been pushed.

        Raises:
            ValueError: If the input tasks and output tasks don't line up.
        """
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
        """
        Iterate through the results.

        Yields:
            ComputeTask: The result of a task.
        """
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
    """Represents a lazy view on a batch tensor and offers various ways of
    accessing the tensor of individual tasks or the whole batch. A batch tensor
    combines multiple rows (tasks). A lazy value can be created in various ways
    and read in various ways. Reading and writing methods do not have to match
    and the data will be converted between them as needed. The lazy value
    needs to be converted to compute values first, though."""
    def __init__(
            self,
            *,
            values_uniform: torch.Tensor | None = None,
            values_padded: tuple[torch.Tensor, list[list[int]]] | None = None,
            values_masked: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> None:
        """
        Creates a lazy value. Exactly one input argument needs to be passed.

        Args:
            values_uniform (torch.Tensor | None, optional): Creates from a
                uniform batch tensor. The first dimension represents the rows
                (tasks). All rows need to have the same shape. Defaults to
                None.
            values_padded (tuple[torch.Tensor, list[list[int]]] | None,
                optional): Creates from a padded tensor. A padded tensor is
                a tuple of the padded tensor (of maximum shape) and a list of
                each individual shape. The first dimension of the tensor
                represents the rows (tasks). Defaults to None.
            values_masked (tuple[torch.Tensor, torch.Tensor] | None, optional):
                Creates from a masked tensor. A masked tensor is a tuple of two
                tensors of maximum shape. One is a padded tensor and the other
                is a binary mask defining which values are valid. The first
                dimension of each tensor represents the rows (tasks). Defaults
                to None.

        Raises:
            ValueError: If not exactly one argument is set.
        """
        if sum([
                values_uniform is not None,
                values_padded is not None,
                values_masked is not None]) != 1:
            raise ValueError("only one argument allowed")
        self._values_uniform = values_uniform
        self._values_padded = values_padded
        self._values_masked = values_masked

    def to_compute_values(self, data_info: DataInfo) -> ComputeValues:
        """
        Converts the lazy value to compute values to make the data readable
        and able to convert into each other.

        Args:
            data_info (DataInfo): The expected data layout.

        Returns:
            ComputeValues: The compute values.
        """
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
        """
        Computes the variable shape. For dimensions that are the same for
        every task, the actual size is returned. For all dimensions that differ
        between tasks, None is returned. However, for masked values, the
        maximum shape is returned instead.

        Returns:
            list[int | None]: The shape.
        """
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
