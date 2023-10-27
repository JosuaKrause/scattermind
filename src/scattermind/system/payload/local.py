import threading

from scattermind.system.base import DataId, L_LOCAL, Locality
from scattermind.system.payload.data import DataStore


class LocalDataId(DataId):
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        try:
            int(raw_id)
        except ValueError:
            return False
        return True


class LocalDataStore(DataStore):
    def __init__(self, max_size: int) -> None:
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
