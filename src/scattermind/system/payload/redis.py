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
"""A redis based payload data storage."""
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
    """The data id for the redis payload data storage. The id is based on a
    hash of the data stored."""
    @staticmethod
    def validate_id(raw_id: str) -> bool:
        return len(raw_id) == bytes_hash_size()


EXPIRE_DEFAULT = 60 * 60.0  # TODO: make configurable
"""Default expiration time for the time base cache freeing strategy."""


class RedisDataStore(DataStore):
    """A redis based payload data store."""
    def __init__(self, cfg: RedisConfig, mode: DataMode) -> None:
        """
        Creates a redis based payload data storage.

        Args:
            cfg (RedisConfig): The redis connection settings.
            mode (DataMode): The strategy used for freeing the cache.
        """
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
        self._redis.set_value(
            key.to_parseable(),
            bytes_to_redis(data),
            expire_in=expire_in)
        return key

    def get_data(self, data_id: DataId) -> bytes | None:
        rdata_id = self.ensure_id_type(data_id, RedisDataId)
        return maybe_redis_to_bytes(
            self._redis.get_value(rdata_id.to_parseable()))

    def data_id_type(self) -> type[RedisDataId]:
        return RedisDataId
