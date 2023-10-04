

import threading
from io import BytesIO, SEEK_END, SEEK_SET

from scattermind.system.readonly.access import ReadonlyAccess
from scattermind.system.readonly.writer import RoAWriter


class RAMAccess(ReadonlyAccess[str], RoAWriter[str]):
    def __init__(self) -> None:
        super().__init__()
        self._objs: dict[str, BytesIO] = {}
        self._lock = threading.RLock()

    @staticmethod
    def is_local_only() -> bool:
        return True

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
