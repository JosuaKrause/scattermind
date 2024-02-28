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
"""Provides a limited implementation of a dictionary that can be properly
sub-classed."""
from collections.abc import Iterable
from typing import Generic, overload, TypeVar


T = TypeVar('T')


class DictHelper(Generic[T]):
    """
    A sub-classable dictionary class with string keys.

    Args:
        Generic (_type_): The value type.
    """
    def __init__(self, obj: dict[str, T] | None = None) -> None:
        self._obj = {} if obj is None else obj

    def set(self, /, key: str, value: T) -> None:
        """
        Set the value for the given key.

        Args:
            key (str): The key.
            value (T): The value.
        """
        self._obj[key] = value

    @overload
    def get(self, /, key: str, default: T) -> T:
        ...

    @overload
    def get(self, /, key: str, default: None) -> T | None:
        ...

    def get(self, /, key: str, default: T | None = None) -> T | None:
        """
        Return the value associated with the key or the default value if the
        key does not exist.

        Args:
            key (str): The key.
            default (T | None, optional): The default value. Defaults to None.

        Returns:
            T | None: The value associated with the key or the default value.
        """
        return self._obj.get(key, default)

    def __getitem__(self, key: str) -> T:
        return self._obj[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._obj[key] = value

    def items(self) -> Iterable[tuple[str, T]]:
        """
        Iterates through all items in the dictionary.

        Yields:
            tuple[str, T]: Key value pairs.
        """
        yield from self._obj.items()

    def keys(self) -> Iterable[str]:
        """
        Iterates through all keys.

        Yields:
            str: The key.
        """
        yield from self._obj.keys()

    def values(self) -> Iterable[T]:
        """
        Iterates through all values.

        Yields:
            T: The value.
        """
        yield from self._obj.values()

    def __str__(self) -> str:
        return self._obj.__str__()

    def __repr__(self) -> str:
        return self._obj.__repr__()
