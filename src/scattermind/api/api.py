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
"""Provides functionality to send tasks and retrieve results."""
import time
from collections.abc import Iterable
from typing import Any, TypedDict

import numpy as np
import torch

from scattermind.system.base import QueueId, TaskId
from scattermind.system.graph.graphdef import FullGraphDefJSON
from scattermind.system.names import GNamespace, QualifiedNodeName
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    ResponseObject,
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TaskStatus,
)
from scattermind.system.torch_util import (
    create_tensor,
    DTypeName,
    str_to_tensor,
)


QueueCounts = TypedDict('QueueCounts', {
    "id": QueueId,
    "name": QualifiedNodeName,
    "queue_length": int,
    "listeners": int,
})
"""Information about a queue."""


class ScattermindAPI:
    """An interface to start tasks and retrieve results."""
    def load_graph(self, graph_def: FullGraphDefJSON) -> GNamespace:
        """
        Load the full graph from a JSON definition.

        Args:
            graph_def (FullGraphDefJSON): The JSON definition of the graph.

        Returns:
            GNamespace: The namespace of the new graph.
        """
        raise NotImplementedError()

    def enqueue(self, ns: GNamespace, value: TaskValueContainer) -> TaskId:
        """
        Enqueues a task. ::py::method:`enqueue_task` provides a more
        user-friendly way of creating a task.

        Args:
            ns (GNamespace): The namespace.
            value (TaskValueContainer): The task's input values.

        Returns:
            TaskId: The task id.
        """
        raise NotImplementedError()

    def get_namespace(self, task_id: TaskId) -> GNamespace | None:
        """
        Retrieves the namespace of the given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            GNamespace | None: The namespace or None if the task does not
                exist.
        """
        raise NotImplementedError()

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """
        Get the status of the given task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            TaskStatus: The status of the task.
        """
        raise NotImplementedError()

    def get_result(self, task_id: TaskId) -> TaskValueContainer | None:
        """
        Retrieve the results of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            TaskValueContainer | None: The results of the task or None if no
                results are available.
        """
        raise NotImplementedError()

    def get_response(self, task_id: TaskId) -> ResponseObject:
        """
        Retrieve the response of a task. The response gives a summary of
        various elements of a task.

        Args:
            task_id (TaskId): The task id.

        Returns:
            ResponseObject: The response.
        """
        raise NotImplementedError()

    def clear_task(self, task_id: TaskId) -> None:
        """
        Remove all data associated with the task.

        Args:
            task_id (TaskId): The task id.
        """
        raise NotImplementedError()

    def enqueue_task(
            self,
            ns: GNamespace | str,
            obj: dict[str, str | list[Any] | np.ndarray | torch.Tensor],
            ) -> TaskId:
        """
        Enqueues a task.

        Args:
            ns (GNamespace | str): The namespace or a namespace string.
            obj (dict[str, str | list[Any] | np.ndarray | torch.Tensor]):
                The task's input values. Values can be strings or various forms
                of tensor data (nested float lists, numpy arrays, etc.).

        Returns:
            TaskId: The task id.
        """
        if not isinstance(ns, GNamespace):
            ns = GNamespace(ns)

        def convert(
                val: str | list[Any] | np.ndarray | torch.Tensor,
                ) -> torch.Tensor:
            if isinstance(val, str):
                return str_to_tensor(val).clone().detach()
            if isinstance(val, torch.Tensor):
                return val.clone().detach()
            return create_tensor(val, dtype=None).clone().detach()

        return self.enqueue(
            ns,
            TaskValueContainer({
                key: convert(value)
                for key, value in obj.items()
            }))

    def wait_for(
            self,
            task_ids: list[TaskId],
            *,
            timeinc: float = 1.0,
            timeout: float | None = 10.0,
            ) -> Iterable[tuple[TaskId, ResponseObject]]:
        """
        Wait for a collection of tasks to complete.

        Args:
            task_ids (list[TaskId]): The tasks to wait for.
            timeinc (float, optional): The increment of internal waiting
                between checks. Defaults to 1.0.
            timeout (float | None, optional): The maximum time to wait for any
                task to complete. If None, no timeout is enforced. Defaults to
                10.0.

        Yields:
            tuple[TaskId, ResponseObject]: Whenever the next task finishes.
                If the wait times out all remaining tasks are returned (which
                will have an in-progress status). A tuple of task id and its
                response.
        """
        assert timeinc > 0.0
        assert timeout is None or timeout > 0.0
        cur_ids = list(task_ids)
        already: set[TaskId] = set()
        start_time = time.monotonic()
        while cur_ids:
            task_id = cur_ids.pop(0)
            status = self.get_status(task_id)
            if status in (
                    TASK_STATUS_READY,
                    TASK_STATUS_DONE,
                    TASK_STATUS_ERROR):
                yield (task_id, self.get_response(task_id))
                start_time = time.monotonic()
                continue
            cur_ids.append(task_id)
            if task_id not in already:
                already.add(task_id)
                continue
            already.clear()
            elapsed = time.monotonic() - start_time
            if timeout is not None and elapsed >= timeout:
                break
            if timeout is not None and elapsed + timeinc > timeout:
                wait_time = timeout - elapsed
            else:
                wait_time = timeinc
            if wait_time > 0.0:
                time.sleep(wait_time)
        for task_id in cur_ids:  # FIXME write timeout test?
            yield (task_id, self.get_response(task_id))

    def namespaces(self) -> set[GNamespace]:
        """
        Retrieve all registered namespaces.

        Returns:
            set[GNamespace]: All namespaces.
        """
        raise NotImplementedError()

    def entry_graph_name(self, ns: GNamespace) -> str:
        """
        Retrieves the name of the entry graph.

        Args:
            ns (GNamespace): The namespace.

        Returns:
            str: The graph name.
        """
        raise NotImplementedError()

    def main_inputs(self, ns: GNamespace) -> set[str]:
        """
        Retrieves the inputs of the main graph.

        Args:
            ns (GNamespace): The namespace.

        Returns:
            set[str]: The names of the input fields.
        """
        raise NotImplementedError()

    def main_outputs(self, ns: GNamespace) -> set[str]:
        """
        Retrieves the outputs of the main graph.

        Args:
            ns (GNamespace): The namespace.

        Returns:
            set[str]: The names of the output fields.
        """
        raise NotImplementedError()

    def output_format(
            self,
            ns: GNamespace,
            output_name: str) -> tuple[DTypeName, list[int | None]]:
        """
        Retrieves the shape of a given main graph output.

        Args:
            ns (GNamespace): The namespace
            output_name (str): The name of the output field.

        Returns:
            tuple[DTypeName, list[int | None]]: The dtype and shape of the
                output.
        """
        raise NotImplementedError()

    def get_queue_stats(self) -> Iterable[QueueCounts]:
        """
        Retrieves information about all active queues.

        Returns:
            Iterable[QueueCounts]: The information about each queue.
        """
        raise NotImplementedError()

    def get_healthcheck(self) -> tuple[str, str, int] | None:
        """
        Gets the address at which a healthcheck is exposed.

        Returns:
            tuple[str, str, int] | None: The in address, out address, port
                tuple. If None, no healthcheck is available.
        """
        raise NotImplementedError()
