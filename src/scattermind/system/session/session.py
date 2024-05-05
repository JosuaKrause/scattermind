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
import os
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import IO

from scattermind.system.base import Module, SessionId, UserId
from scattermind.system.io import (
    ensure_folder,
    listdir,
    open_readb,
    open_reads,
    open_writeb,
    open_writes,
    remove_file,
)
from scattermind.system.util import get_file_hash, get_time_str, parse_time_str


class Session:
    def __init__(
            self,
            sessions: 'SessionStore',
            session_id: SessionId) -> None:
        self._sid = session_id
        self._sessions = sessions

    def get_session_id(self) -> SessionId:
        return self._sid

    def set_value(self, key: str, value: str | None) -> None:
        self._sessions.set_value(self._sid, key, value)

    def get_value(self, key: str) -> str | None:
        return self._sessions.get_value(self._sid, key)

    def local_folder(self) -> str:
        return self._sessions.local_folder(self._sid)

    def clear_local(self) -> None:
        self._sessions.clear_local(self._sid)

    def sync_in(self) -> None:
        self._sessions.sync_in(self._sid)

    def sync_out(self) -> None:
        self._sessions.sync_out(self._sid)

    def remove(self) -> None:
        self._sessions.remove(self._sid)


class SessionStore(Module):
    def __init__(self, *, cache_path: str) -> None:
        self._cache_path = ensure_folder(cache_path)

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        raise NotImplementedError()

    def get_session(self, session_id: SessionId) -> Session:
        return Session(self, session_id)

    def set_value(
            self, session_id: SessionId, key: str, value: str | None) -> None:
        raise NotImplementedError()

    def get_value(self, session_id: SessionId, key: str) -> str | None:
        raise NotImplementedError()

    @contextmanager
    def open_blob_write(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        raise NotImplementedError()

    @contextmanager
    def open_blob_read(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        raise NotImplementedError()

    def blob_hash(self, session_id: SessionId, name: str) -> str:
        raise NotImplementedError()

    def blob_list(self, session_id: SessionId) -> list[str]:
        raise NotImplementedError()

    def blob_remove(self, session_id: SessionId, name: str) -> None:
        raise NotImplementedError()

    def local_folder(self, session_id: SessionId) -> str:
        return os.path.join(self._cache_path, *session_id.as_folder())

    def locals(self) -> Iterable[SessionId]:
        yield from SessionId.find_folder_ids(self._cache_path)

    def clear_local(self, session_id: SessionId) -> None:
        path = self.local_folder(session_id)
        shutil.rmtree(path, ignore_errors=True)

    def local_time(self, session_id: SessionId) -> datetime | None:
        path = self.local_folder(session_id)
        try:
            with open_reads(os.path.join(path, ".time")) as fin:
                return parse_time_str(fin.read(200).strip())
        except FileNotFoundError:
            return None

    def sync_in(self, session_id: SessionId) -> None:
        path = self.local_folder(session_id)
        blobs: set[str] = set(self.blob_list(session_id))
        need_copy: set[str] = set(blobs)
        for fname in listdir(path):
            full_path = os.path.join(path, fname)
            if fname not in blobs:
                remove_file(full_path)
                continue
            own_hash = get_file_hash(full_path)
            in_hash = self.blob_hash(session_id, fname)
            if own_hash == in_hash:
                need_copy.discard(fname)
        for fname in need_copy:
            full_path = os.path.join(path, fname)
            with open_writeb(full_path) as fout:
                with self.open_blob_read(session_id, fname) as fin:
                    shutil.copyfileobj(fin, fout)  # type: ignore
        with open_writes(os.path.join(path, ".time")) as fout:
            print(get_time_str(), file=fout)

    def sync_out(self, session_id: SessionId) -> None:
        path = self.local_folder(session_id)
        local: set[str] = {
            fname
            for fname in listdir(path)
            if not fname.startswith(".")
        }
        need_copy: set[str] = set(local)
        for fname in self.blob_list(session_id):
            if fname not in local:
                self.blob_remove(session_id, fname)
                continue
            full_path = os.path.join(path, fname)
            other_hash = self.blob_hash(session_id, fname)
            out_hash = get_file_hash(full_path)
            if other_hash == out_hash:
                need_copy.discard(fname)
        for fname in need_copy:
            full_path = os.path.join(path, fname)
            with self.open_blob_write(session_id, fname) as fout:
                with open_readb(full_path) as fin:
                    shutil.copyfileobj(fin, fout)  # type: ignore

    def remove(self, session_id: SessionId) -> None:
        raise NotImplementedError()
