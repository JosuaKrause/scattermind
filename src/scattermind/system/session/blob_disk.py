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
"""A session blob storage using the local file system and redis."""
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import IO

from redipy import Redis, RedisConfig

from scattermind.system.base import L_EITHER, Locality, SessionId
from scattermind.system.io import ensure_folder, open_readb, open_writeb
from scattermind.system.session.session import SessionBlob
from scattermind.system.util import get_file_hash


class DiskSessionBlob(SessionBlob):
    """A session blob storage using the local file system and redis."""
    def __init__(self, cfg: RedisConfig, *, disk_path: str) -> None:
        self._redis = Redis("redis", cfg=cfg, redis_module="session")
        self._disk_path = disk_path

    def locality(self) -> Locality:
        return L_EITHER

    @staticmethod
    def _fname_key(session_id: SessionId, fname: str) -> str:
        return f"fname:{session_id.to_parseable()}:{fname}"

    def _get_hashpath(self, hash_str: str, *, ensure: bool = True) -> str:
        full_path = os.path.join(
            self._disk_path,
            hash_str[:2],
            hash_str[2:4],
            f"{hash_str[4:]}.blob")
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
        names = self.blob_list(session_id)
        with self._redis.pipeline() as pipe:
            pipe.delete(*(self._fname_key(session_id, name) for name in names))
