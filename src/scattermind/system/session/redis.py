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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import IO, TypeVar

from redipy import Redis, RedisConfig

from scattermind.system.base import L_EITHER, Locality, SessionId, UserId
from scattermind.system.io import ensure_folder, open_readb, open_writeb
from scattermind.system.session.session import Session, SessionStore
from scattermind.system.util import get_file_hash


T = TypeVar('T')


class RedisSessionStore(SessionStore):
    """
    A session storage using redis and the local file system.
    """
    def __init__(
            self,
            cfg: RedisConfig,
            *,
            disk_path: str,
            cache_path: str,
            is_shared: bool) -> None:
        super().__init__(cache_path=cache_path, is_shared=is_shared)
        self._redis = Redis("redis", cfg=cfg, redis_module="session")
        self._disk_path = disk_path

    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    @staticmethod
    def _fname_key(session_id: SessionId, fname: str) -> str:
        return f"fname:{session_id.to_parseable()}:{fname}"

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
            condition: Callable[[], T],
            *,
            timeout: float) -> T | None:
        return self._redis.wait_for(
            self._signal_key(session_id, key), condition, timeout)

    def _get_hashpath(self, hash_str: str, *, ensure: bool = True) -> str:
        full_path = os.path.join(
            self._disk_path,
            hash_str[:2],
            f"{hash_str[2:]}.blob")
        if ensure:
            ensure_folder(os.path.dirname(full_path))
            return full_path
        return full_path

    @contextmanager
    def open_blob_write(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        key = self._fname_key(session_id, name)

        def get_file(key: str, tmp: str) -> str:
            hash_str = get_file_hash(tmp)
            self._redis.set_value(key, hash_str)
            print(f"writing {hash_str=} {key=} {name=}")
            return self._get_hashpath(hash_str)

        with open_writeb(
                key,
                tmp_base=self._disk_path,
                filename_fn=get_file) as fout:
            yield fout

    @contextmanager
    def open_blob_read(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        key = self._fname_key(session_id, name)
        hash_str = self._redis.get_value(key)
        if hash_str is None:
            raise FileNotFoundError(
                f"cannot find file for {session_id=} {name=}")
        with open_readb(self._get_hashpath(hash_str, ensure=False)) as fin:
            yield fin

    def blob_hash(
            self,
            session_id: SessionId,
            names: Iterable[str]) -> dict[str, str]:
        res: dict[str, str] = {}
        names = list(names)
        with self._redis.pipeline() as pipe:
            for name in names:
                key = self._fname_key(session_id, name)
                pipe.get_value(key)
            hashes = pipe.execute()
            print(names, hashes)
            for name, hash_str in zip(names, hashes):
                if hash_str is None:
                    raise FileNotFoundError(
                        f"cannot find file for {session_id=} {name=}")
                res[name] = hash_str
        return res

    def blob_list(self, session_id: SessionId) -> list[str]:
        prefix = self._fname_key(session_id, "")
        return sorted({
            key.removeprefix(prefix)
            for key in self._redis.iter_keys(match=f"{prefix}*")
        })

    def blob_remove(self, session_id: SessionId, names: list[str]) -> None:
        self._redis.delete(*(
            self._fname_key(session_id, name)
            for name in names
        ))

    def remove(self, session_id: SessionId) -> None:
        # FIXME: make redis function
        user_id = self.get_user(session_id)
        if user_id is None:
            raise ValueError(f"cannot find user of {session_id=}")
        keys = self.get_keys(session_id)
        names = self.blob_list(session_id)
        with self._redis.pipeline() as pipe:
            pipe.delete(self._stu_key(session_id))
            pipe.srem(self._uts_key(user_id), session_id.to_parseable())
            pipe.delete(*(self._val_key(session_id, key) for key in keys))
            pipe.delete(*(self._fname_key(session_id, name) for name in names))
        self.clear_local(session_id)
