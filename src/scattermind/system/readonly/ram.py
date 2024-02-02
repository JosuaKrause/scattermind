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
"""A RAM based readonly access. It has a writer interface initially fill the
data. It is most commonly used for test cases."""
import threading
from io import BytesIO, SEEK_END, SEEK_SET

from scattermind.system.base import L_EITHER, Locality
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.readonly.writer import RoAWriter


class RAMAccess(ReadonlyAccess[str], RoAWriter[str]):
    """A RAM based readonly access. It has a writer interface initially fill
    the data. It is most commonly used for test cases."""
    def __init__(self) -> None:
        super().__init__()
        self._objs: dict[str, BytesIO] = {}
        self._lock = threading.RLock()

    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    def open_raw(self, path: str) -> str:
        return path

    def read_raw(self, hnd: str, offset: int, length: int) -> bytes:
        with self._lock:
            fout = self._objs[hnd]
            fout.seek(offset, SEEK_SET)
            return fout.read(length)

    def open_write_raw(self, path: str) -> str:
        with self._lock:
            buff = self._objs.get(path)
            if buff is None:
                self._objs[path] = BytesIO()
            return path

    def write_raw(self, hnd: str, data: bytes) -> tuple[int, int]:
        with self._lock:
            fin = self._objs[hnd]
            offset = fin.seek(0, SEEK_END)
            length = fin.write(data)
            return offset, length

    def close(self, hnd: str) -> None:
        pass
