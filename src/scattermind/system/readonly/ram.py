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
"""A RAM based readonly access. It has a writer interface initially fill the
data. It is most commonly used for test cases."""
import os
import threading
from io import BytesIO, SEEK_END, SEEK_SET

from scattermind.system.base import L_EITHER, Locality, NodeId
from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.readonly.writer import RoAWriter


class RAMAccess(ReadonlyAccess[str], RoAWriter[str]):
    """A RAM based readonly access. It has a writer interface initially fill
    the data. It is most commonly used for test cases."""
    def __init__(self, scratch: str) -> None:
        super().__init__()
        self._objs: dict[str, BytesIO] = {}
        self._scratch_folder = scratch
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

    def do_get_scratchspace(self, node_id: NodeId) -> str:
        name = node_id.to_parseable()
        prefix = name[:3]
        postfix = name[3:]
        path = os.path.join(self._scratch_folder, prefix, postfix)
        os.makedirs(path, exist_ok=True)
        return path
