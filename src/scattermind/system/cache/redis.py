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
from typing import Literal

from redipy import Redis, RedisConfig

from scattermind.system.base import CacheId, L_EITHER, Locality, TaskId
from scattermind.system.cache.cache import GraphCache
from scattermind.system.info import DataFormat
from scattermind.system.logger.log import EventStream
from scattermind.system.payload.data import DataStore
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.queue.queue import QueuePool
from scattermind.system.redis_util import redis_to_tvc, tvc_to_redis


RedisPrefix = Literal["data", "listeners"]
DATA_PREFIX: RedisPrefix = "data"
LISTENERS_PREFIX: RedisPrefix = "listeners"


class RedisCache(GraphCache):
    """Cache using redis."""
    def __init__(self, cfg: RedisConfig, *, use_defer: bool) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="cache")
        self._use_defer = use_defer

    def locality(self) -> Locality:
        return L_EITHER

    @staticmethod
    def key(prefix: RedisPrefix, cache_id: CacheId) -> str:
        """
        Computes the full key.

        Args:
            prefix (RedisPrefix): The key prefix.
            cache_id (CacheId): The cache id.

        Returns:
            str: The full key.
        """
        return f"{prefix}:{cache_id.to_parseable()}"

    def put_cached_output(
            self,
            logger: EventStream,
            store: DataStore,
            queue_pool: QueuePool,
            *,
            cache_id: CacheId,
            output_data: TaskValueContainer) -> None:
        listeners_key = self.key(LISTENERS_PREFIX, cache_id)
        with self._redis.pipeline() as pipe:
            pipe.set_value(
                self.key(DATA_PREFIX, cache_id), tvc_to_redis(output_data))
            pipe.smembers(listeners_key)
            pipe.delete(listeners_key)
            _, listeners, _ = pipe.execute()
        for listener in listeners:
            task_id = TaskId.parse(listener)
            queue_pool.maybe_requeue_task_id(
                logger, store, task_id, error_info=None)

    def put_progress(self, cache_id: CacheId, task_id: TaskId) -> None:
        if not self._use_defer:
            return
        self._redis.set_value(
            self.key(DATA_PREFIX, cache_id), task_id.to_parseable())

    def add_listener(self, cache_id: CacheId, listener_id: TaskId) -> None:
        listeners_key = self.key(LISTENERS_PREFIX, cache_id)
        with self._redis.pipeline() as pipe:
            pipe.get_value(self.key(DATA_PREFIX, cache_id))
            pipe.sadd(listeners_key, listener_id.to_parseable())
            output, _ = pipe.execute()
        base_task = None
        if output and output[0] == TaskId.prefix():
            base_task = TaskId.parse(output)
        if base_task is None:
            self._redis.delete(listeners_key)

    def get_cached_output(
            self,
            cache_id: CacheId,
            output_format: DataFormat) -> TaskValueContainer | TaskId | None:
        res = self._redis.get_value(self.key(DATA_PREFIX, cache_id))
        if res is None:
            return None
        if res and res[0] == TaskId.prefix():
            return TaskId.parse(res)
        return redis_to_tvc(res, output_format)
