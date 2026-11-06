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
"""Tests removing sessions that hold no keys or blobs."""
import tempfile
from typing import Any

import pytest
from redipy import Redis

from scattermind.system.base import SessionId, UserId
from scattermind.system.names import UName
from scattermind.system.redis_util import create_redis, get_test_config
from scattermind.system.session.blob_disk import DiskSessionBlob
from scattermind.system.session.kv_redis import RedisSessionKeyValue


class StrictRedis:
    """
    Wraps a redis client and rejects `DEL` without keys like a real redis
    server does ("wrong number of arguments for 'del' command"). The in-memory
    backend silently accepts empty deletes.
    """
    def __init__(self, redis: Any) -> None:
        self._redis = redis

    def delete(self, *keys: str) -> int:
        """
        Deletes keys. Raises if there are none.

        Args:
            *keys (str): The keys.

        Returns:
            int: The number of deleted keys.
        """
        if not keys:
            raise ValueError(
                "ERR wrong number of arguments for 'del' command")
        return self._redis.delete(*keys)

    def pipeline(self) -> Any:
        """
        Creates a strict pipeline.

        Returns:
            Any: A context manager yielding a strict pipeline.
        """
        return _StrictPipeContext(self._redis.pipeline())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._redis, name)


class _StrictPipeContext:
    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __enter__(self) -> StrictRedis:
        return StrictRedis(self._ctx.__enter__())

    def __exit__(self, *args: Any) -> Any:
        return self._ctx.__exit__(*args)


def _make_redis(is_redis: bool) -> StrictRedis:
    if is_redis:
        return StrictRedis(
            create_redis(get_test_config(), redis_module="session"))
    return StrictRedis(Redis("memory"))


def _new_session_id() -> SessionId:
    return SessionId.create_for(UserId.create(UName("test")))


@pytest.mark.parametrize("is_redis", [False, True])
def test_remove_empty_kv(is_redis: bool) -> None:
    """
    Test that removing a session without keys does not call `DEL` without
    arguments.

    Args:
        is_redis (bool): Whether to use redis.
    """
    store = RedisSessionKeyValue(get_test_config())
    # pylint: disable=protected-access
    store._redis = _make_redis(is_redis)  # type: ignore
    session_id = _new_session_id()
    assert not store.get_keys(session_id)
    store.remove(session_id)  # must not raise
    store.push_value(session_id, "a", "1")
    assert store.get_keys(session_id) == ["a"]
    store.remove(session_id)
    assert not store.get_keys(session_id)


@pytest.mark.parametrize("is_redis", [False, True])
def test_remove_empty_blob(is_redis: bool) -> None:
    """
    Test that removing a session or blobs without any blobs does not call
    `DEL` without arguments.

    Args:
        is_redis (bool): Whether to use redis.
    """
    with tempfile.TemporaryDirectory() as disk_path:
        store = DiskSessionBlob(get_test_config(), disk_path=disk_path)
        # pylint: disable=protected-access
        store._redis = _make_redis(is_redis)  # type: ignore
        session_id = _new_session_id()
        assert not store.blob_list(session_id)
        store.blob_remove(session_id, [])  # must not raise
        store.remove(session_id)  # must not raise
        with store.open_blob_write(session_id, "f.txt") as fout:
            fout.write(b"hello")
        assert store.blob_list(session_id) == ["f.txt"]
        store.blob_remove(session_id, ["f.txt"])
        assert not store.blob_list(session_id)
        with store.open_blob_write(session_id, "g.txt") as fout:
            fout.write(b"hello")
        store.remove(session_id)
        assert not store.blob_list(session_id)
