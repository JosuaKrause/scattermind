from redipy import Redis, RedisConfig

from scattermind.system.base import DataId, L_REMOTE, Locality
from scattermind.system.payload.data import DataStore
from scattermind.system.redis_util import (
    bytes_to_redis,
    DataMode,
    maybe_redis_to_bytes,
)
from scattermind.system.util import bytes_hash_size, get_bytes_hash


class RedisDataId(DataId):
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        return len(raw_id) == bytes_hash_size()


class RedisDataStore(DataStore):
    def __init__(self, cfg: RedisConfig, mode: DataMode) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="data")
        self._mode = mode  # TODO: implement time mode

    def _generate_redis_id(self, value: bytes) -> RedisDataId:
        return RedisDataId.parse(
            f"{RedisDataId.prefix()}{get_bytes_hash(value)}")

    @staticmethod
    def locality() -> Locality:
        return L_REMOTE

    @staticmethod
    def is_content_addressable() -> bool:
        return True

    def store_data(self, data: bytes) -> DataId:
        key = self._generate_redis_id(data)
        self._redis.set(key.to_parseable(), bytes_to_redis(data))
        return key

    def get_data(self, data_id: DataId) -> bytes | None:
        if not isinstance(data_id, RedisDataId):
            raise ValueError(
                f"unexpected {data_id.__class__.__name__}: {data_id}")
        return maybe_redis_to_bytes(self._redis.get(data_id.to_parseable()))
