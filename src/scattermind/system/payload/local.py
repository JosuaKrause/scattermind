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
"""An in-memory payload data store."""
import threading

from scattermind.system.base import DataId, L_LOCAL, Locality
from scattermind.system.payload.data import DataStore


class LocalDataId(DataId):
    """The data id type for the in-memory payload data store. The id is from
    a simple integer sequence."""
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        try:
            int(raw_id)
        except ValueError:
            return False
        return True


class LocalDataStore(DataStore):
    """An in-memory payload data store. Ids are assigned via integer sequence.
    When the maximum number of entries is reached the oldest (smallest integer
    value) ids get purged first."""
    def __init__(self, max_size: int) -> None:
        """
        Create a local payload data store.

        Args:
            max_size (int): The maximum number of entries.
        """
        super().__init__()
        self._size = 0
        self._max_size = max_size
        self._idcount = 0
        self._data: dict[LocalDataId, bytes] = {}
        self._lock = threading.RLock()

    def _generate_data_id(self) -> LocalDataId:
        with self._lock:
            data_id = LocalDataId.parse(
                f"{LocalDataId.prefix()}{self._idcount}")
            self._idcount += 1
            return data_id

    def _gc(self) -> None:
        with self._lock:
            keys = sorted(
                self._data.keys(), key=lambda data_id: int(data_id.raw_id()))
            while keys and self._size > self._max_size:
                cur = keys.pop(0)
                data = self._data.pop(cur, None)
                if data is not None:
                    self._size -= len(data)

    @staticmethod
    def locality() -> Locality:
        return L_LOCAL

    @staticmethod
    def is_content_addressable() -> bool:
        return False

    def store_data(self, data: bytes) -> LocalDataId:
        with self._lock:
            data_id = self._generate_data_id()
            self._data[data_id] = data
            self._size += len(data)
            self._gc()
        return data_id

    def get_data(self, data_id: DataId) -> bytes | None:
        return self._data.get(self.ensure_id_type(data_id, LocalDataId))

    def data_id_type(self) -> type[LocalDataId]:
        return LocalDataId
