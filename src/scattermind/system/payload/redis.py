from redipy import Redis, RedisConfig

from scattermind.system.base import DataId, L_REMOTE, Locality
from scattermind.system.payload.data import DataStore
from scattermind.system.util import (
    as_base85,
    bytes_hash_size,
    from_base85,
    get_bytes_hash,
)


class RedisDataId(DataId):
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        return len(raw_id) == bytes_hash_size()


class RedisDataStore(DataStore):
    def __init__(self, cfg: RedisConfig) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="data")

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
        self._redis.set(key.to_parseable(), as_base85(data))
        return key

    def get_data(self, data_id: DataId) -> bytes | None:
        if not isinstance(data_id, RedisDataId):
            raise ValueError(
                f"unexpected {data_id.__class__.__name__}: {data_id}")
        res = self._redis.get(data_id.to_parseable())
        if res is None:
            return None
        return from_base85(res)
