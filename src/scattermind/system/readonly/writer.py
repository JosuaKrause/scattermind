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
