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
"""A writer access is a write interface for a readonly access. In normal
operation only the readonly interface is exposed. The write interface is used
to fill the data of the access. This happens most often in offline scripts
that are run when setting up an execution graph."""
import contextlib
from collections.abc import Iterator
from typing import Generic, TypeVar

import torch

from scattermind.system.torch_util import serialize_tensor


T = TypeVar('T')


class WriterHandle(Generic[T]):
    """A writer handle to add data to a handle."""
    def __init__(self, roaw: 'RoAWriter', path: str, hnd: T) -> None:
        """
        Creates a handle with write access.

        Args:
            roaw (RoAWriter): The underlying writer access.
            path (str): The key of the blob.
            hnd (T): The implementation specific handle.
        """
        self._roaw = roaw
        self._path = path
        self._hnd = hnd

    def write(self, data: bytes) -> tuple[int, int]:
        """
        Appends data at the current write location.

        Args:
            data (bytes): The data to append.

        Returns:
            tuple[int, int]: The offset and length at which the data can be
                accessed.
        """
        return self._roaw.write_raw(self._hnd, data)

    def write_tensor(self, value: torch.Tensor) -> tuple[int, int]:
        """
        Append a tensor at the current write location.

        Args:
            value (torch.Tensor): The tensor.

        Returns:
            tuple[int, int]: The offset and length at which the tensor can be
                accessed.
        """
        return self._roaw.write_tensor(self._hnd, value)

    def as_data_str(self, pos: tuple[int, int]) -> str:
        """
        Convert a write location (such as the result of a write operation on
        the handle) to a string that can be used in an execution graph
        definition JSON.

        Args:
            pos (tuple[int, int]): The write location (offset and length).

        Returns:
            str: The location identifier to be used in an execution graph file.
        """
        offset, length = pos
        return f"{offset}:{length}:{self._path}"

    def close(self) -> None:
        """Closes the write handler."""
        self._roaw.close(self._hnd)


class RoAWriter(Generic[T]):
    """A writer access is a write interface for a readonly access. In normal
    operation only the readonly interface is exposed. The write interface is
    used to fill the data of the access. This happens most often in offline
    scripts that are run when setting up an execution graph. The writer is
    assumed to write once, i.e., the data to write should be ready and laid out
    already when writing to the access. Writing is append only. That means,
    once data has been written it cannot be overwritten. All write operations
    return the location of the data for later read access. The location should
    be converted to a location identifier, using the appropriate handler
    method, in order to use it in an execution graph definition JSON file.
    """
    @contextlib.contextmanager
    def open_write(self, path: str) -> Iterator[WriterHandle[T]]:
        """
        Open a handle for writing.

        Args:
            path (str): The key of the blob.

        Yields:
            WriterHandle[T]: The handle.
        """
        try:
            hnd = WriterHandle(self, path, self.open_write_raw(path))
            yield hnd
        finally:
            hnd.close()

    def write_tensor(
            self,
            hnd: T,
            value: torch.Tensor) -> tuple[int, int]:
        """
        Appends a tensor to the end of the given handle.

        Args:
            hnd (T): The handle.
            value (torch.Tensor): The tensor.

        Returns:
            tuple[int, int]: The location (offset, length) for reading.
        """
        return self.write_raw(hnd, serialize_tensor(value))

    def open_write_raw(self, path: str) -> T:
        """
        Opens a raw handle for writing.

        Args:
            path (str): The key of the blob.

        Returns:
            T: The implementation defined handle.
        """
        raise NotImplementedError()

    def write_raw(self, hnd: T, data: bytes) -> tuple[int, int]:
        """
        Append to a raw handle.

        Args:
            hnd (T): The implementation defined handle.
            data (bytes): The data to write.

        Returns:
            tuple[int, int]: The location (offset, length) for reading.
        """
        raise NotImplementedError()

    def close(self, hnd: T) -> None:
        """
        Closes the raw handle.

        Args:
            hnd (T): The implementation defined handle.
        """
        raise NotImplementedError()
