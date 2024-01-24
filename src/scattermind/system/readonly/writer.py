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
import contextlib
from collections.abc import Iterator
from typing import Generic, TypeVar

import torch

from scattermind.system.torch_util import serialize_tensor


T = TypeVar('T')


class WriterHandle(Generic[T]):
    def __init__(self, roaw: 'RoAWriter', path: str, hnd: T) -> None:
        self._roaw = roaw
        self._path = path
        self._hnd = hnd

    def write(self, data: bytes) -> tuple[int, int]:
        return self._roaw.write_raw(self._hnd, data)

    def write_tensor(self, value: torch.Tensor) -> tuple[int, int]:
        return self._roaw.write_tensor(self._hnd, value)

    def as_data_str(self, pos: tuple[int, int]) -> str:
        offset, length = pos
        return f"{offset}:{length}:{self._path}"

    def close(self) -> None:
        self._roaw.close(self._hnd)


class RoAWriter(Generic[T]):
    @contextlib.contextmanager
    def open_write(self, path: str) -> Iterator[WriterHandle[T]]:
        try:
            hnd = WriterHandle(self, path, self.open_write_raw(path))
            yield hnd
        finally:
            hnd.close()

    def write_tensor(
            self,
            hnd: T,
            value: torch.Tensor) -> tuple[int, int]:
        return self.write_raw(hnd, serialize_tensor(value))

    def open_write_raw(self, path: str) -> T:
        raise NotImplementedError()

    def write_raw(self, hnd: T, data: bytes) -> tuple[int, int]:
        raise NotImplementedError()

    def close(self, hnd: T) -> None:
        raise NotImplementedError()
