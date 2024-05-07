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
"""A session storage using redis and the local file system.
"""
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import IO

from redipy import Redis, RedisConfig

from scattermind.system.base import L_EITHER, Locality, SessionId, UserId
from scattermind.system.io import (
    ensure_folder,
    listdir,
    open_readb,
    open_writeb,
    remove_file,
)
from scattermind.system.session.session import Session, SessionStore
from scattermind.system.util import get_blob_hash


class RedisSessionStore(SessionStore):
    """
    A session storage using redis and the local file system.
    """
    def __init__(
            self,
            cfg: RedisConfig,
            *,
            disk_path: str,
            cache_path: str) -> None:
        super().__init__(cache_path=cache_path)
        self._redis = Redis("redis", cfg=cfg, redis_module="session")
        self._disk_path = disk_path

    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    @staticmethod
    def _uts_key(user_id: UserId) -> str:
        return f"uts:{user_id.to_parseable()}"

    @staticmethod
    def _stu_key(session_id: SessionId) -> str:
        return f"stu:{session_id.to_parseable()}"

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

    def register_session(self, user_id: UserId, session_id: SessionId) -> None:
        with self._redis.pipeline() as pipe:
            pipe.sadd(self._uts_key(user_id), session_id.to_parseable())
            pipe.set_value(self._stu_key(session_id), user_id.to_parseable())

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        user_sessions = self._redis.smembers(self._uts_key(user_id))
        yield from (
            self.get_session(SessionId.parse(sess_str))
            for sess_str in user_sessions
        )

    def get_user(self, session_id: SessionId) -> UserId | None:
        user_str = self._redis.get_value(self._stu_key(session_id))
        return None if user_str is None else UserId.parse(user_str)

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
            condition: Callable[[], bool],
            timeout: float) -> bool:
        res = self._redis.wait_for(
            self._signal_key(session_id, key), condition, timeout)
        return bool(res)

    def _get_folder(
            self, session_id: SessionId, *, ensure: bool = True) -> str:
        folder = os.path.join(self._disk_path, *session_id.as_folder())
        if ensure:
            return ensure_folder(folder)
        return folder

    def _get_path(self, session_id: SessionId, name: str) -> str:
        if not name or name.startswith(".") or "/" in name:
            raise ValueError(f"invalid {name=} for {session_id=}")
        return os.path.join(self._get_folder(session_id), name)

    @contextmanager
    def open_blob_write(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        with open_writeb(self._get_path(session_id, name)) as fout:
            yield fout

    @contextmanager
    def open_blob_read(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        with open_readb(self._get_path(session_id, name)) as fin:
            yield fin

    def blob_hash(self, session_id: SessionId, name: str) -> str:
        with self.open_blob_read(session_id, name) as fin:
            return get_blob_hash(fin)

    def blob_list(self, session_id: SessionId) -> list[str]:
        path = self._get_folder(session_id, ensure=False)
        return [fname for fname in listdir(path) if not fname.startswith(".")]

    def blob_remove(self, session_id: SessionId, name: str) -> None:
        remove_file(self._get_path(session_id, name))

    def remove(self, session_id: SessionId) -> None:
        # FIXME: make redis function
        user_id = self.get_user(session_id)
        if user_id is None:
            raise ValueError(f"cannot find user of {session_id=}")
        keys = self.get_keys(session_id)
        with self._redis.pipeline() as pipe:
            pipe.delete(self._stu_key(session_id))
            pipe.srem(self._uts_key(user_id), session_id.to_parseable())
            pipe.delete(*(self._val_key(session_id, key) for key in keys))
        folder = self._get_folder(session_id, ensure=False)
        shutil.rmtree(folder, ignore_errors=True)
        self.clear_local(session_id)
