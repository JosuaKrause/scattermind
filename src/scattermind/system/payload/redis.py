from redipy import Redis, RedisConfig

from scattermind.system.base import DataId, L_REMOTE, Locality
from scattermind.system.payload.data import DataStore
from scattermind.system.redis_util import (
    bytes_to_redis,
    DataMode,
    DM_TIME,
    maybe_redis_to_bytes,
)
from scattermind.system.util import bytes_hash_size, get_bytes_hash


class RedisDataId(DataId):
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        return len(raw_id) == bytes_hash_size()


EXPIRE_DEFAULT = 60 * 60.0  # TODO: make configurable


class RedisDataStore(DataStore):
    def __init__(self, cfg: RedisConfig, mode: DataMode) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="data")
        self._mode = mode

    def _generate_redis_id(self, value: bytes) -> RedisDataId:
        return RedisDataId.parse(
            f"{RedisDataId.prefix()}{get_bytes_hash(value)}")

    @staticmethod
    def locality() -> Locality:
        return L_REMOTE

    @staticmethod
    def is_content_addressable() -> bool:
        return True

    def store_data(self, data: bytes) -> RedisDataId:
        key = self._generate_redis_id(data)
        expire_in = None
        if self._mode == DM_TIME:
            expire_in = EXPIRE_DEFAULT
        self._redis.set(
            key.to_parseable(),
            bytes_to_redis(data),
            expire_in=expire_in)
        return key

    def get_data(self, data_id: DataId) -> bytes | None:
        rdata_id = self.ensure_id_type(data_id, RedisDataId)
        return maybe_redis_to_bytes(self._redis.get(rdata_id.to_parseable()))

    def data_id_type(self) -> type[RedisDataId]:
        return RedisDataId
