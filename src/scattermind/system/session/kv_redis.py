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
"""A session key value storage using redis."""
from collections.abc import Callable
from typing import TypeVar

from redipy import Redis, RedisConfig

from scattermind.system.base import L_EITHER, Locality, SessionId
from scattermind.system.session.session import SessionKeyValue


T = TypeVar('T')


class RedisSessionKeyValue(SessionKeyValue):
    """A session key value storage using redis."""
    def __init__(self, cfg: RedisConfig) -> None:
        self._redis = Redis("redis", cfg=cfg, redis_module="session")

    def locality(self) -> Locality:
        return L_EITHER

    @staticmethod
    def _val_key(session_id: SessionId, key: str | None) -> str:
        if key is None:
            key = ""
        elif not key:
            raise ValueError(f"invalid {key=}")
        return f"val:{session_id.to_parseable()}:{key}"

    @staticmethod
    def _signal_key(session_id: SessionId, key: str) -> str:
        if not key:
            raise ValueError(f"invalid {key=}")
        return f"signal:{session_id.to_parseable()}:{key}"

    def set_value(
            self,
            session_id: SessionId,
            key: str,
            index: int,
            value: str) -> None:
        # FIXME: implement as redis function
        redis = self._redis
        val_key = self._val_key(session_id, key)
        if index == redis.llen(val_key):
            self.push_value(session_id, key, value)
            return
        redis.lset(val_key, index, value)

    def push_value(self, session_id: SessionId, key: str, value: str) -> None:
        self._redis.rpush(self._val_key(session_id, key), value)

    def pop_value(self, session_id: SessionId, key: str) -> str | None:
        return self._redis.rpop(self._val_key(session_id, key))

    def get_value(
            self, session_id: SessionId, key: str, index: int) -> str | None:
        return self._redis.lindex(self._val_key(session_id, key), index)

    def get_length(self, session_id: SessionId, key: str) -> int:
        return self._redis.llen(self._val_key(session_id, key))

    def get_keys(self, session_id: SessionId) -> list[str]:
        key_prefix = self._val_key(session_id, None)
        return [
            res.removeprefix(key_prefix)
            for res in self._redis.keys(match=f"{key_prefix}*", block=False)
        ]

    def notify_signal(self, session_id: SessionId, key: str) -> None:
        self._redis.publish(self._signal_key(session_id, key), "signal")

    def wait_for_signal(
            self,
            session_id: SessionId,
            key: str,
            condition: Callable[[], T],
            *,
            timeout: float) -> T | None:
        return self._redis.wait_for(
            self._signal_key(session_id, key), condition, timeout)

    def remove(self, session_id: SessionId) -> None:
        # FIXME: make redis function
        keys = self.get_keys(session_id)
        with self._redis.pipeline() as pipe:
            pipe.delete(*(self._val_key(session_id, key) for key in keys))
