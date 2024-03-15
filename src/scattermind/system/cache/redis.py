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
"""A caching layer implemented in redis."""
from redipy import Redis, RedisConfig

from scattermind.system.base import CacheId, L_EITHER, Locality
from scattermind.system.cache.cache import GraphCache
from scattermind.system.info import DataFormat
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.redis_util import redis_to_tvc, tvc_to_redis


class RedisCache(GraphCache):
    """Cache using redis."""
    def __init__(self, cfg: RedisConfig) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="cache")

    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    @staticmethod
    def key(cache_id: CacheId) -> str:
        """
        Computes the full key.

        Args:
            cache_id (CacheId): The cache id.

        Returns:
            str: The full key.
        """
        return f"{cache_id.to_parseable()}"

    def put_cached_output(
            self,
            cache_id: CacheId,
            output_data: TaskValueContainer) -> None:
        self._redis.set_value(self.key(cache_id), tvc_to_redis(output_data))

    def get_cached_output(
            self,
            cache_id: CacheId,
            output_format: DataFormat) -> TaskValueContainer | None:
        res = self._redis.get_value(self.key(cache_id))
        if res is None:
            return None
        return redis_to_tvc(res, output_format)
