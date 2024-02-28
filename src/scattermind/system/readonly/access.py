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
"""The main interface for `readonly` data access."""
import contextlib
from collections.abc import Iterator
from typing import Generic, TYPE_CHECKING, TypeVar

import torch

from scattermind.system.base import Module, NodeId
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor


if TYPE_CHECKING:
    from scattermind.system.graph.node import Node


T = TypeVar('T')


DataAccess = tuple[str, int, int]
"""Tuple specifying a chunk of data. The string is the name of the blob,
the first integer is the offset within the blob, and the second integer is
the length to be read from that offset."""


class RoAHandle(Generic[T]):
    """A readonly handle to a blob."""
    def __init__(self, roa: 'ReadonlyAccess', hnd: T) -> None:
        """
        Creates a handle for a blob.

        Args:
            roa (ReadonlyAccess): The underlying readonly access.
            hnd (T): The implementation specific handle that allows access
            to the blob.
        """
        self._roa = roa
        self._hnd = hnd

    def read(self, offset: int, length: int) -> bytes:
        """
        Reads a chunk of data from the blob.

        Args:
            offset (int): The offset within the blob.
            length (int): The length to read.

        Returns:
            bytes: The bytes read.
        """
        return self._roa.read_raw(self._hnd, offset, length)

    def read_tensor(
            self,
            offset: int,
            length: int,
            data_info: DataInfo) -> torch.Tensor:
        """
        Read a tensor from the given location in the blob.

        Args:
            offset (int): The offset within the blob.
            length (int): The length of the data chunk.
            data_info (DataInfo): Expected information about the tensor.

        Returns:
            torch.Tensor: The tensor.
        """
        return self._roa.read_tensor(self._hnd, offset, length, data_info)

    def close(self) -> None:
        """Close the handle for the blob access."""
        self._roa.close(self._hnd)


class ReadonlyAccess(Module, Generic[T]):
    """
    Provides readonly access in a key value store way. Each blob associated
    with a key can be read via random access. The data is meant to be
    unchangeable (hence `readonly`) and is most commonly used for
    stored weights.
    """
    @contextlib.contextmanager
    def open(self, path: str) -> Iterator[RoAHandle[T]]:
        """
        Opens a handle to the blob associated with the given key.

        Args:
            path (str): The key.

        Yields:
            RoAHandle[T]: The handle to access the contents of the blob.
        """
        try:
            hnd = RoAHandle(self, self.open_raw(path))
            yield hnd
        finally:
            hnd.close()

    def load_tensor(
            self, data: DataAccess, data_info: DataInfo) -> torch.Tensor:
        """
        Loads a tensor from a chunk of data.

        Args:
            data (DataAccess): The full location of the data.
            data_info (DataInfo): The expected information about the tensor.

        Returns:
            torch.Tensor: The tensor.
        """
        path, offset, length = data
        with self.open(path) as hnd:
            return hnd.read_tensor(offset, length, data_info)

    def read_tensor(
            self,
            hnd: T,
            offset: int,
            length: int,
            data_info: DataInfo) -> torch.Tensor:
        """
        Read a tensor from a given chunk.

        Args:
            hnd (T): The handle.
            offset (int): The offset of the chunnk.
            length (int): The length of the chunk.
            data_info (DataInfo): The expected information about the tensor.

        Returns:
            torch.Tensor: The tensor.
        """
        data = self.read_raw(hnd, offset, length)
        res = deserialize_tensor(data, data_info.dtype())
        return data_info.check_tensor(res)

    def open_raw(self, path: str) -> T:
        """
        Opens a raw handle for the given blob key.

        Args:
            path (str): The blob key.

        Returns:
            T: The implementation defined handle.
        """
        raise NotImplementedError()

    def read_raw(self, hnd: T, offset: int, length: int) -> bytes:
        """
        Read bytes from a location with a raw handle.

        Args:
            hnd (T): The implementation defined handle.
            offset (int): The offset of the chunk.
            length (int): The length of the chunk.

        Returns:
            bytes: The bytes read.
        """
        raise NotImplementedError()

    def close(self, hnd: T) -> None:
        """
        Closes a raw handle.

        Args:
            hnd (T): The implementation defined handle.
        """
        raise NotImplementedError()

    def get_scratchspace(self, node: 'Node') -> str:
        """
        Creates a local temporary folder location for a given node. The
        scratchspace can be written to and read from and the data persists.
        The data can be removed by the system at any time if the node is not
        loaded. If multiple executors load the same node no synchronization
        happens and the node needs to make sure to handle concurrent write
        access.

        Args:
            node (Node): The node.

        Returns:
            str: The local path to the folder.
        """
        return self.do_get_scratchspace(node.get_id())

    def do_get_scratchspace(self, node_id: NodeId) -> str:
        """
        Creates a local temporary folder location for a given node. The
        scratchspace can be written to and read from and the data persists.
        The data can be removed by the system at any time if the node is not
        loaded. If multiple executors load the same node no synchronization
        happens and the node needs to make sure to handle concurrent write
        access.

        This is the internal implementation. Call
        ::py:method:`get_scratchspace` instead.

        Args:
            node_id (NodeId): The node id.

        Returns:
            str: The local path to the folder.
        """
        raise NotImplementedError()
