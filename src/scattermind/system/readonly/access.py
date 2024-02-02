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
"""The main interface for `readonly` data access."""
import contextlib
from collections.abc import Iterator
from typing import Generic, TypeVar

import torch

from scattermind.system.base import Module
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor


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
