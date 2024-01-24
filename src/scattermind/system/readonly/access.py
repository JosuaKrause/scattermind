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

from scattermind.system.base import Module
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor


T = TypeVar('T')


DataAccess = tuple[str, int, int]


class RoAHandle(Generic[T]):
    def __init__(self, roa: 'ReadonlyAccess', hnd: T) -> None:
        self._roa = roa
        self._hnd = hnd

    def read(self, offset: int, length: int) -> bytes:
        return self._roa.read_raw(self._hnd, offset, length)

    def read_tensor(
            self,
            offset: int,
            length: int,
            data_info: DataInfo) -> torch.Tensor:
        return self._roa.read_tensor(self._hnd, offset, length, data_info)

    def close(self) -> None:
        self._roa.close(self._hnd)


class ReadonlyAccess(Module, Generic[T]):
    @contextlib.contextmanager
    def open(self, path: str) -> Iterator[RoAHandle[T]]:
        try:
            hnd = RoAHandle(self, self.open_raw(path))
            yield hnd
        finally:
            hnd.close()

    def load_tensor(
            self, data: DataAccess, data_info: DataInfo) -> torch.Tensor:
        path, offset, length = data
        with self.open(path) as hnd:
            return hnd.read_tensor(offset, length, data_info)

    def read_tensor(
            self,
            hnd: T,
            offset: int,
            length: int,
            data_info: DataInfo) -> torch.Tensor:
        data = self.read_raw(hnd, offset, length)
        res = deserialize_tensor(data, data_info.dtype())
        return data_info.check_tensor(res)

    def open_raw(self, path: str) -> T:
        raise NotImplementedError()

    def read_raw(self, hnd: T, offset: int, length: int) -> bytes:
        raise NotImplementedError()

    def close(self, hnd: T) -> None:
        raise NotImplementedError()
