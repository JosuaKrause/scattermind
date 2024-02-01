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
"""Provides functionality to send tasks and retrieve results."""
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from scattermind.system.base import TaskId
from scattermind.system.graph.graphdef import FullGraphDefJSON
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.response import (
    ResponseObject,
    TASK_STATUS_DONE,
    TASK_STATUS_ERROR,
    TASK_STATUS_READY,
    TaskStatus,
)
from scattermind.system.torch_util import create_tensor, str_to_tensor


class ScattermindAPI:
    """An interface to start tasks and retrieve results."""
    def load_graph(self, graph_def: FullGraphDefJSON) -> None:
        """
        Load the full graph from a JSON definition.

        Args:
            graph_def (FullGraphDefJSON): The JSON definition of the graph.
        """
        raise NotImplementedError()

    def enqueue(self, value: TaskValueContainer) -> TaskId:
        """
        Enqueues a task. ::py::method:`enqueue_task` provides a more
        user-friendly way of creating a task.

        Args:
            value (TaskValueContainer): The task's input values.

        Returns:
            TaskId: The task id.
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
            obj: dict[str, str | list[Any] | np.ndarray | torch.Tensor],
            ) -> TaskId:
        """
        Enqueues a task.

        Args:
            obj (dict[str, str | list[Any] | np.ndarray | torch.Tensor]):
                The task's input values. Values can be strings or various forms
                of tensor data (nested float lists, numpy arrays, etc.).

        Returns:
            TaskId: The task id.
        """

        def convert(
                val: str | list[Any] | np.ndarray | torch.Tensor,
                ) -> torch.Tensor:
            if isinstance(val, str):
                return str_to_tensor(val).clone().detach()
            if isinstance(val, torch.Tensor):
                return val.clone().detach()
            return create_tensor(val, dtype=None).clone().detach()

        return self.enqueue(TaskValueContainer({
            key: convert(value)
            for key, value in obj.items()
        }))

    def wait_for(
            self,
            task_ids: list[TaskId],
            *,
            timeinc: float = 1.0,
            timeout: float = 10.0,
            ) -> Iterable[tuple[TaskId, ResponseObject]]:
        """
        Wait for a collection of tasks to complete.

        Args:
            task_ids (list[TaskId]): The tasks to wait for.
            timeinc (float, optional): The increment of internal waiting
                between checks. Defaults to 1.0.
            timeout (float, optional): The maximum time to wait for any task
                to complete. Defaults to 10.0.

        Yields:
            tuple[TaskId, ResponseObject]: Whenever the next task finishes.
                If the wait times out all remaining tasks are returned (which
                will have an in-progress status). A tuple of task id and its
                response.
        """
        assert timeinc > 0.0
        assert timeout > 0.0
        cur_ids = list(task_ids)
        already: set[TaskId] = set()
        start_time = time.monotonic()
        while cur_ids:
            task_id = cur_ids.pop(0)
            status = self.get_status(task_id)
            print(f"{ctx_fmt()} wait for {task_id} {status}")
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
            if elapsed >= timeout:
                break
            if elapsed + timeinc > timeout:
                wait_time = timeout - elapsed
            else:
                wait_time = timeinc
            if wait_time > 0.0:
                time.sleep(wait_time)
        for task_id in cur_ids:  # FIXME write timeout test?
            yield (task_id, self.get_response(task_id))
