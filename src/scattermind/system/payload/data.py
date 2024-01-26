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
"""Interface for payload data storing."""
from typing import TypeVar

import torch

from scattermind.system.base import DataId, Module
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor, serialize_tensor


DT = TypeVar('DT', bound=DataId)
"""The `DataId` subclass understood by a given `DataStore` implementation."""


class DataStore(Module):
    """A data store handling payload data which gets produced during graph
    execution. Unlike the original input data, data from intermediate
    and final results, aka payload data, is stored in volatile storage. This
    allows for graceful handling of overload. Implementations of the data store
    are free to choose under which circumstances payload data will be freed
    (e.g., after a certain amount of time or when running low on memory). Every
    consumer of payload data must handle the case where data is no longer
    available. By default, if payload data is missing for a task, the task
    should be requeued anew (using the non-volatile input data) and the retries
    counter should be increased by one. If the retry counter grows above a
    certain threshold an error should be emitted instead.

    Payload data is identified via data id."""
    def store_tensor(self, value: torch.Tensor) -> DataId:
        """
        Stores a tensor.

        Args:
            value (torch.Tensor): The tensor.

        Returns:
            DataId: The corresponding data id.
        """
        return self.store_data(serialize_tensor(value))

    def get_tensor(
            self,
            data_id: DataId,
            data_info: DataInfo) -> torch.Tensor | None:
        """
        Retrieves the tensor associated with the given data id.

        Args:
            data_id (DataId): The data id.
            data_info (DataInfo): The expected tensor information.

        Returns:
            torch.Tensor | None: The associated tensor or None if the data
                was freed.
        """
        data = self.get_data(data_id)
        if data is None:
            return None
        res = deserialize_tensor(data, data_info.dtype())
        return data_info.check_tensor(res)

    def ensure_id_type(self, data_id: DataId, data_id_type: type[DT]) -> DT:
        """
        Verifies that the given data id has a compatible format to the store.

        Args:
            data_id (DataId): The data id.
            data_id_type (type[DT]): The expected data id type.

        Raises:
            ValueError: If the data id is not compatible.

        Returns:
            DT: The data id as the expected data id type.
        """
        if not isinstance(data_id, data_id_type):
            raise ValueError(
                f"unexpected {data_id.__class__.__name__}: {data_id}")
        return data_id

    @staticmethod
    def is_content_addressable() -> bool:
        """
        Whether the storage uses (the hashed) content of the stored data as id.
        Using a content addressable store can be beneficial when storing the
        same data frequently (lower memory usage). Additionally, content
        addressable data ids can directly be used for caching node outputs
        (if all input ids are the same and the node is deterministic
        (i.e., pure) the cached output id can be returned without computing
        the node).

        See also :py:method::`scattermind.system.graph.node.Node#is_pure`.

        Returns:
            bool: True, if the store is content addressable.
        """
        raise NotImplementedError()

    def store_data(self, data: bytes) -> DataId:
        """
        Stores payload data.

        Args:
            data (bytes): The payload data.

        Returns:
            DataId: The corresponding data id
        """
        raise NotImplementedError()

    def get_data(self, data_id: DataId) -> bytes | None:
        """
        Retrieves the payload data associated with the given data id.

        Args:
            data_id (DataId): The data id.

        Returns:
            bytes | None: The associated payload data or None if the data
                was freed.
        """
        raise NotImplementedError()

    def data_id_type(self) -> type[DataId]:
        """
        The concrete subclass of `DataId` compatible with this storage
        implementation.

        Returns:
            type[DataId]: The data id associated with this storage.
        """
        raise NotImplementedError()
