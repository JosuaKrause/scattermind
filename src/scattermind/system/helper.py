from collections.abc import Iterable
from typing import Generic, overload, TypeVar


T = TypeVar('T')


class DictHelper(Generic[T]):
    def __init__(self, obj: dict[str, T] | None = None) -> None:
        self._obj = {} if obj is None else obj

    def set(self, /, key: str, value: T) -> None:
        self._obj[key] = value

    @overload
    def get(self, /, key: str, default: T) -> T:
        ...

    @overload
    def get(self, /, key: str, default: None) -> T | None:
        ...

    def get(self, /, key: str, default: T | None = None) -> T | None:
        return self._obj.get(key, default)

    def __getitem__(self, key: str) -> T:
        return self._obj[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._obj[key] = value

    def items(self) -> Iterable[tuple[str, T]]:
        yield from self._obj.items()

    def keys(self) -> Iterable[str]:
        yield from self._obj.keys()

    def values(self) -> Iterable[T]:
        yield from self._obj.values()

    def __str__(self) -> str:
        return self._obj.__str__()

    def __repr__(self) -> str:
        return self._obj.__repr__()
