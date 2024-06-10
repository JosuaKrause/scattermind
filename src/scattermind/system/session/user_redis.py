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
"""A session user storage using redis."""
from collections.abc import Iterable
from typing import TypeVar

from redipy import Redis, RedisConfig

from scattermind.system.base import L_EITHER, Locality, SessionId, UserId
from scattermind.system.session.session import (
    Session,
    SessionStore,
    SessionUser,
)


T = TypeVar('T')


class RedisSessionUser(SessionUser):
    """A session user storage using redis."""
    def __init__(self, cfg: RedisConfig) -> None:
        self._redis = Redis("redis", cfg=cfg, redis_module="session")

    def locality(self) -> Locality:
        return L_EITHER

    @staticmethod
    def _uts_key(user_id: UserId) -> str:
        return f"uts:{user_id.to_parseable()}"

    @staticmethod
    def _stu_key(session_id: SessionId) -> str:
        return f"stu:{session_id.to_parseable()}"

    def register_session(self, user_id: UserId, session_id: SessionId) -> None:
        with self._redis.pipeline() as pipe:
            pipe.sadd(self._uts_key(user_id), session_id.to_parseable())
            pipe.set_value(self._stu_key(session_id), user_id.to_parseable())

    def get_sessions(
            self,
            sessions: SessionStore,
            user_id: UserId) -> Iterable[Session]:
        user_sessions = self._redis.smembers(self._uts_key(user_id))
        yield from (
            sessions.get_session(SessionId.parse(sess_str))
            for sess_str in user_sessions
        )

    def get_user(self, session_id: SessionId) -> UserId | None:
        user_str = self._redis.get_value(self._stu_key(session_id))
        return None if user_str is None else UserId.parse(user_str)

    def remove(self, session_id: SessionId) -> None:
        # FIXME: make redis function
        user_id = self.get_user(session_id)
        if user_id is None:
            raise ValueError(f"cannot find user of {session_id=}")
        with self._redis.pipeline() as pipe:
            pipe.delete(self._stu_key(session_id))
            pipe.srem(self._uts_key(user_id), session_id.to_parseable())
