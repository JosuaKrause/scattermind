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
from collections.abc import Iterable
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch

from scattermind.system.base import QueueId, SessionId, TaskId, UserId
from scattermind.system.graph.graphdef import FullGraphDefJSON
from scattermind.system.info import DataFormat
from scattermind.system.names import GNamespace, QualifiedNodeName, UName
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.redis_util import (
    redis_to_robj,
    redis_to_tensor,
    robj_to_redis,
    tensor_to_redis,
)
from scattermind.system.response import ResponseObject, TaskStatus
from scattermind.system.session.session import Session
from scattermind.system.torch_util import (
    create_tensor,
    DTypeName,
    str_to_tensor,
)


InputTypes: TypeAlias = (
    str
    | Session
    | SessionId
    | list[Any]
    | np.ndarray
    | torch.Tensor)
"""Types for task inputs."""


QueueCounts = TypedDict('QueueCounts', {
    "id": QueueId,
    "name": QualifiedNodeName,
    "queue_length": int,
    "listeners": int,
})
"""Information about a queue."""


def convert_data_to_tensor(val: InputTypes) -> torch.Tensor:
    """
    Convert standard data types into their tensor representation.

    Args:
        val (InputTypes): The value to convert.

    Returns:
        torch.Tensor: The tensor.
    """
    if isinstance(val, SessionId):
        return val.to_tensor().clone().detach()
    if isinstance(val, Session):
        return val.get_session_id().to_tensor().clone().detach()
    if isinstance(val, str):
        return str_to_tensor(val).clone().detach()
    if isinstance(val, torch.Tensor):
        return val.clone().detach()
    return create_tensor(val, dtype=None).clone().detach()


def task_input_to_redis(obj: dict[str, InputTypes]) -> str:
    """
    Convert task inputs into a redis storable format.

    Args:
        obj (dict[str, InputTypes]): The task inputs.

    Returns:
        str: The redis storable format.
    """
    return robj_to_redis({
        key: tensor_to_redis(convert_data_to_tensor(value))
        for key, value in obj.items()
    })


def redis_to_task_input(
        text: str, input_format: DataFormat) -> dict[str, InputTypes]:
    """
    Convert a redis value to a task input.

    Args:
        text (str): The redis value.
        input_format (DataFormat): The expected data format of the task.

    Returns:
        dict[str, InputTypes]: The task input.
    """
    obj = redis_to_robj(text)
    return {
        key: redis_to_tensor(value, input_format[key].dtype())
        for key, value in obj.items()
    }


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

    def enqueue(
            self,
            ns: GNamespace,
            value: TaskValueContainer,
            *,
            task_id: TaskId | None = None) -> TaskId:
        """
        Enqueues a task. ::py::method:`enqueue_task` provides a more
        user-friendly way of creating a task.

        Args:
            ns (GNamespace): The namespace.
            value (TaskValueContainer): The task's input values.
            task_id (TaskId | None, optional): The task id to use for the task.
                If set, the user has to ensure that the id is globally unique.

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
        Get the status of the given task. Note, calling this method is
        required before being able to access results.

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
            obj: dict[str, InputTypes],
            *,
            task_id: TaskId | None = None,
            ) -> TaskId:
        """
        Enqueues a task.

        Args:
            ns (GNamespace | str): The namespace or a namespace string.
            obj (dict[str, InputTypes]):
                The task's input values. Values can be strings or various forms
                of tensor data (nested float lists, numpy arrays, etc.) and
                other special types.
            task_id (TaskId | None, optional): The task id to use for the task.
                If set, the user has to ensure that the id is globally unique.

        Returns:
            TaskId: The task id.
        """
        if not isinstance(ns, GNamespace):
            ns = GNamespace(ns)
        return self.enqueue(
            ns,
            TaskValueContainer({
                key: convert_data_to_tensor(value)
                for key, value in obj.items()
            }),
            task_id=task_id)

    def wait_for(
            self,
            task_ids: list[TaskId],
            *,
            timeout: float | None = 10.0,
            auto_clear: bool = False,
            ) -> Iterable[tuple[TaskId, ResponseObject]]:
        """
        Wait for a collection of tasks to complete.

        Args:
            task_ids (list[TaskId]): The tasks to wait for.
            timeout (float | None, optional): The maximum time to wait for any
                task to complete. The timeout is reset after each successfully
                returned task. If None, no timeout is enforced. Defaults to
                10.0.
            auto_clear (bool, optional): If True, tasks get automatically
                cleared after they have been processed. Note, that this will
                also clear tasks that have timed out. After the loop finishes
                no task in the list will be valid anymore. Defaults to False.

        Yields:
            tuple[TaskId, ResponseObject]: Whenever the next task finishes.
                If the wait times out all remaining tasks are returned (which
                will have an in-progress status). A tuple of task id and its
                response.
        """
        raise NotImplementedError()

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
        inputs = self.main_input_format(ns)
        return set(inputs.keys())

    def main_input_format(self, ns: GNamespace) -> DataFormat:
        """
        Retrieves the input format of the main graph.

        Args:
            ns (GNamespace): The namespace.

        Returns:
            DataFormat: The data format.
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

    def get_queue_stats(
            self, ns: GNamespace | None = None) -> Iterable[QueueCounts]:
        """
        Retrieves information about all active queues.

        Args:
            ns (GNamespace | None, optional): The namespace to filter. Defaults
                to no filter.

        Returns:
            Iterable[QueueCounts]: The information about each queue.
        """
        raise NotImplementedError()

    def has_any_tasks(self) -> bool:
        """
        Checks whether any tasks are present in any queue.

        Returns:
            bool: True, if there are some tasks to be computed.
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

    def get_user_id(self, user_name: str) -> UserId:
        """
        Get the user id for the given name.

        Args:
            user_name (str): The user name.

        Returns:
            UserId: The user id.
        """
        return UserId.create(UName(user_name))

    def new_session(
            self,
            user_id: UserId,
            *,
            copy_from: Session | None = None) -> Session:
        """
        Create a new session for the given user.

        Args:
            user_id (UserId): The user id.

            copy_from (Session | None): If not None the new session is
                initialized with values from a different session.

        Returns:
            Session: The session.
        """
        raise NotImplementedError()

    def get_session(self, session_id: SessionId) -> Session:
        """
        Returns a session object for the given session id.

        Args:
            session_id (SessionId): The session id.

        Returns:
            Session: The session object.
        """
        raise NotImplementedError()

    def get_session_user(self, session_id: SessionId) -> UserId | None:
        """
        Retrieves the user associated with the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            UserId | None: The user or None if the session is not associated
                with any user (e.g., when it was deleted).
        """
        raise NotImplementedError()

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        """
        Returns all sessions associated with the given user.

        Args:
            user_id (UserId): The user id.

        Yields:
            Session: A session object.
        """
        raise NotImplementedError()
