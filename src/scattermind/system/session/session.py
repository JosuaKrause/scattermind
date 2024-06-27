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
"""Defines the session interface that allows direct access to ongoing tasks and
large user specific blobs. If further allows to retain information across
multiple tasks.
"""
import contextlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import IO, TypedDict, TypeVar

from scattermind.system.base import (
    L_EITHER,
    Locality,
    Module,
    SessionId,
    UserId,
)
from scattermind.system.io import (
    ensure_folder,
    get_files,
    open_readb,
    open_reads,
    open_writeb,
    open_writes,
    remove_file,
    TMP_POSTFIX,
)
from scattermind.system.util import (
    get_file_hash,
    maybe_fmt_time,
    maybe_parse_time_str,
    now,
)


T = TypeVar('T')


INFO_NAME = ".info"
"""The name of the local info file."""


InfoObj = TypedDict('InfoObj', {
    "time_in": datetime | None,
    "time_out": datetime | None,
})
"""The info file content. "time_in" is the time of the last incoming sync and
"time_out" is the time of the last outgoing sync (if any)."""


PROCESS_LOCK = threading.RLock()
"""The lock for generating process ids."""


PROCESS_ID: str | None = None
"""The unique process id."""


def get_process_id() -> str:
    """
    Get a unique id for this process.

    Returns:
        str: The id for this process.
    """
    global PROCESS_ID  # pylint: disable=global-statement

    if PROCESS_ID is None:
        with PROCESS_LOCK:
            if PROCESS_ID is None:
                # FIXME: make it not completely random every time
                PROCESS_ID = uuid.uuid4().hex
    return PROCESS_ID


class Session:
    """A session enables direct access to a (running) task. There are three
    modes of direct access: notifications from a task, bi-directional immediate
    value access, and long term storage. Notifications can be emitted from a
    task when certain milestones are reached. Value access happens via arrays.
    Here, changes are immediately visible to the other side. Long term storage
    is a way to retain state across multiple tasks (e.g., associating a new
    task with an existing session id allows access to previously cached values
    or the history of its inputs). During computation the latest content of the
    long term storage is only visible to the executing task. Upon node
    execution completion the content is made accessible."""
    def __init__(
            self,
            sessions: 'SessionStore',
            session_id: SessionId) -> None:
        self._sid = session_id
        self._sessions = sessions

    def get_session_id(self) -> SessionId:
        """
        The session id.

        Returns:
            SessionId: The session id.
        """
        return self._sid

    def get_session_store(self) -> 'SessionStore':
        """
        The session store.

        Returns:
            SessionStore: The session store.
        """
        return self._sessions

    def set_value(
            self,
            key: str,
            index: int,
            value: str) -> None:
        """
        Sets the array value at the given index. If the index is equal to the
        length of the array a new value is appended. Negative index values
        address the array from the back. If the index is outside the array and
        it is not the positive length of the array an `IndexError` is raised.

        Raises:
            IndexError: If the index is outside the array and it is not the
                positive length of the array.

        Args:
            key (str): The key.
            index (int): The index.
            value (str): The value to set.
        """
        self._sessions.set_value(self._sid, key, index, value)

    def push_value(self, key: str, value: str) -> None:
        """
        Appends a value to the end of the array.

        Args:
            key (str): The key.
            value (str): The value.
        """
        self._sessions.push_value(self._sid, key, value)

    def pop_value(self, key: str) -> str | None:
        """
        Removes and returns the last value of the array.

        Args:
            key (str): The key.

        Returns:
            str | None: The value at the end of the array or None if the array
                was empty.
        """
        return self._sessions.pop_value(self._sid, key)

    def get_value(self, key: str, index: int) -> str | None:
        """
        Gets the value at the given index of the array. A negative index gets
        the value from the back of the array. Out of bounds access returns
        None.

        Args:
            key (str): The key.
            index (int): The index.

        Returns:
            str | None: The value or None if the index was out of bounds.
        """
        return self._sessions.get_value(self._sid, key, index)

    def get_length(self, key: str) -> int:
        """
        Gets the length of the given array. If the key does not exist 0 is
        returned.

        Args:
            key (str): The key.

        Returns:
            int: The length of the array.
        """
        return self._sessions.get_length(self._sid, key)

    def get_keys(self) -> list[str]:
        """
        Returns all keys of the given session.

        Returns:
            list[str]: The list of all keys.
        """
        return self._sessions.get_keys(self._sid)

    def notify_signal(self, key: str) -> None:
        """
        Emits a signal on the given key. Note, that signal keys are independent
        of value keys.

        Args:
            key (str): The signal key.
        """
        self._sessions.notify_signal(self._sid, key)

    def wait_for_signal(
            self,
            key: str,
            condition: Callable[[], T],
            *,
            timeout: float) -> T | None:
        """
        Waits for a signal on the given key and returns if the condition is
        met. If no signal arrives or the condition is not met within the
        timeout False is returned. Note, that signal keys are independent
        of value keys.

        Args:
            key (str): The signal key

            condition (Callable[[], T]): If the condition can be converted to
                True, the method returns successfully with its result.

            timeout (float): The timeout in seconds.

        Returns:
            T | None: The result of the condition if it could be evaluated to
                True. Otherwise, if the timeout occurred, None is returned.
        """
        return self._sessions.wait_for_signal(
            self._sid, key, condition, timeout=timeout)

    @contextlib.contextmanager
    def get_local_folder(self) -> Iterator[str]:
        """
        Returns the local folder where long term storage can be accessed in a
        resource block. The folder is populated with all content before
        returning the path and upon successful completion of the block the
        content is synchronized. Note, if the block encounters an error no
        synchronization happens. Files starting with '.' are not synchronized.

        Yields:
            str: The path.
        """
        sid = self._sid
        sessions = self._sessions
        local_folder = sessions.local_folder(sid)
        sessions.sync_in(sid)
        yield local_folder
        sessions.sync_out(sid)

    @staticmethod
    @contextlib.contextmanager
    def get_local_folders(session_arr: list['Session']) -> Iterator[list[str]]:
        """
        Returns the local folders where long term storage can be accessed of a
        list of sessions in a resource block. The folders are populated with
        all content before returning the paths and upon successful completion
        of the block the contents are synchronized. If an error is encountered
        during population or execution of the block no synchronization happens.
        If an error happens during synchronization, synchronization continues
        for other sessions. In this case only the first error is returned.
        Files starting with '.' are not synchronized.

        Args:
            session_arr (list[Session]): The sessions.

        Yields:
            list[str]: The paths associated to corresponding sessions.
        """
        if not session_arr:
            yield []
            return
        sessions = session_arr[0].get_session_store()
        sids = [
            session.get_session_id()
            for session in session_arr
            if session.get_session_store() is sessions
        ]
        if len(sids) != len(session_arr):
            raise ValueError(
                f"all sessions must use the same store: {session_arr}")
        success = False

        def sync_out(sid: SessionId) -> BaseException | None:
            # pylint: disable=broad-exception-caught

            if not success:
                return None
            try:
                sessions.sync_out(sid)
            except BaseException as err:
                # TODO: maybe log anyway
                return err
            return None

        local_folders = [sessions.local_folder(sid) for sid in sids]
        try:
            for sid in sids:
                sessions.sync_in(sid)
            yield local_folders
            success = True
        finally:
            exc: BaseException | None = None
            for sid in sids:
                err = sync_out(sid)
                if err is not None and exc is None:
                    exc = err
            if exc is not None:
                raise exc

    def sync_in(self) -> None:
        """
        Populates or updates the local copy of the given session.
        """
        self._sessions.sync_in(self._sid)

    def sync_out(self) -> None:
        """
        Synchronizes the local copy of the given session. This ensures that
        remote blobs are exactly the same as the local copy. If blobs do not
        exist locally anymore they are deleted remotely as well.
        """
        self._sessions.sync_out(self._sid)

    def clear_local(self) -> None:
        """
        Immediately empty the local copy of the session. If called outside of
        a synchronization block only local disk space is freed.
        """
        self._sessions.clear_local(self._sid)

    def remove(self) -> None:
        """
        Removes everything associated with the given session. Note, this does
        not guarantee that all local copies are removed as well.
        """
        self._sessions.remove(self._sid)


class SessionStore(Module):
    """
    A session storage which manages sessions.

    Sessions enable direct access to a (running) task. There are three
    modes of direct access: notifications from a task, bi-directional immediate
    value access, and long term storage. Notifications can be emitted from a
    task when certain milestones are reached. Value access happens via arrays.
    Here, changes are immediately visible to the other side. Long term storage
    is a way to retain state across multiple tasks (e.g., associating a new
    task with an existing session id allows access to previously cached values
    or the history of its inputs). During computation the latest content of the
    long term storage is only visible to the executing task. Upon node
    execution completion the content is made accessible.
    """
    def __init__(
            self,
            session_user: 'SessionUser',
            session_blob: 'SessionBlob',
            session_key_value: 'SessionKeyValue',
            *,
            cache_path: str,
            is_shared: bool) -> None:
        """
        Creates a session store.

        Args:
            session_user (SessionUser): The session user handler.

            session_blob (SessionBlob): The session blob handler.

            session_key_value (SessionKeyValue): The session key value handler.

            cache_path (str): The cache root.

            is_shared (bool): Whether the cache root is shared. If it is shared
                each process gets assigned a random folder in the cache root.
                Otherwise, the cache root is used directly.
        """
        if is_shared:
            cache_path = os.path.join(cache_path, get_process_id())
        self._cache_path = ensure_folder(cache_path)
        self._scratch_path = ensure_folder(
            os.path.join(self._cache_path, "scratch"))
        self._session_user = session_user
        self._session_blob = session_blob
        self._session_key_value = session_key_value

    def locality(self) -> Locality:
        b_loc = self._session_blob.locality()
        kv_loc = self._session_key_value.locality()
        if b_loc == L_EITHER:
            return kv_loc
        if kv_loc == L_EITHER:
            return b_loc
        if b_loc != kv_loc:
            raise ValueError(
                f"mismatching localities for {self._session_blob=} ({b_loc}) "
                f"and {self._session_key_value=} ({kv_loc})")
        return b_loc

    def create_new_session(
            self,
            user_id: UserId,
            *,
            copy_from: SessionId | None = None) -> Session:
        """
        Creates a new session for the given user id. Session ids are guaranteed
        to be unique across all users.

        Args:
            user_id (UserId): The user.

            copy_from (SessionId | None, optional): If not None the content
                of a different session is copied over. Defaults to None.

        Returns:
            Session: The new session.
        """
        res = SessionId.create_for(user_id)
        self.register_session(user_id, res)
        if copy_from is not None:
            self.copy_session_blobs(copy_from, res)
        return self.get_session(res)

    def register_session(self, user_id: UserId, session_id: SessionId) -> None:
        """
        Registers the session so the user and session can be associated.

        Args:
            user_id (UserId): The user.
            session_id (SessionId): The session.
        """
        self._session_user.register_session(user_id, session_id)

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        """
        Retrieves all sessions of the given user.

        Args:
            user_id (UserId): The user.

        Returns:
            Iterable[Session]: The sessions for the user.
        """
        return self._session_user.get_sessions(self, user_id)

    def get_user(self, session_id: SessionId) -> UserId | None:
        """
        Retrieves the user associated with the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            UserId | None: The user or None if the session is not associated
                with any user (e.g., when it was deleted).
        """
        return self._session_user.get_user(session_id)

    def get_session(self, session_id: SessionId) -> Session:
        """
        Gets the session handle for the given id.

        Args:
            session_id (SessionId): The session id.

        Returns:
            Session: The session handle.
        """
        return Session(self, session_id)

    def set_value(
            self,
            session_id: SessionId,
            key: str,
            index: int,
            value: str) -> None:
        """
        Sets the array value at the given index. If the index is equal to the
        length of the array a new value is appended. Negative index values
        address the array from the back. If the index is outside the array and
        it is not the positive length of the array an `IndexError` is raised.

        Raises:
            IndexError: If the index is outside the array and it is not the
                positive length of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            index (int): The index.
            value (str): The value to set.
        """
        self._session_key_value.set_value(session_id, key, index, value)

    def push_value(self, session_id: SessionId, key: str, value: str) -> None:
        """
        Appends a value to the end of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            value (str): The value.
        """
        self._session_key_value.push_value(session_id, key, value)

    def pop_value(self, session_id: SessionId, key: str) -> str | None:
        """
        Removes and returns the last value of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.

        Returns:
            str | None: The value at the end of the array or None if the array
                was empty.
        """
        return self._session_key_value.pop_value(session_id, key)

    def get_value(
            self, session_id: SessionId, key: str, index: int) -> str | None:
        """
        Gets the value at the given index of the array. A negative index gets
        the value from the back of the array. Out of bounds access returns
        None.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            index (int): The index.

        Returns:
            str | None: The value or None if the index was out of bounds.
        """
        return self._session_key_value.get_value(session_id, key, index)

    def get_length(self, session_id: SessionId, key: str) -> int:
        """
        Gets the length of the given array. If the key does not exist 0 is
        returned.

        Args:
            session_id (SessionId): The session.
            key (str): The key.

        Returns:
            int: The length of the array.
        """
        return self._session_key_value.get_length(session_id, key)

    def get_keys(self, session_id: SessionId) -> list[str]:
        """
        Returns all keys of the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            list[str]: The list of all keys.
        """
        return self._session_key_value.get_keys(session_id)

    def notify_signal(self, session_id: SessionId, key: str) -> None:
        """
        Emits a signal on the given key. Note, that signal keys are independent
        of value keys.

        Args:
            session_id (SessionId): The session.
            key (str): The signal key.
        """
        self._session_key_value.notify_signal(session_id, key)

    def wait_for_signal(
            self,
            session_id: SessionId,
            key: str,
            condition: Callable[[], T],
            *,
            timeout: float) -> T | None:
        """
        Waits for a signal on the given key and returns if the condition is
        met. If no signal arrives or the condition is not met within the
        timeout False is returned. Note, that signal keys are independent
        of value keys.

        Args:
            session_id (SessionId): The session.

            key (str): The signal key

            condition (Callable[[], T]): If the condition can be converted to
                True, the method returns successfully with its result.

            timeout (float): The timeout in seconds.

        Returns:
            T | None: The result of the condition if it could be evaluated to
                True. Otherwise, if the timeout occurred, None is returned.
        """
        return self._session_key_value.wait_for_signal(
            session_id, key, condition, timeout=timeout)

    @contextmanager
    def open_blob_write(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        """
        Opens a blob for writing. This method is used for synchronization and
        should not be normally used. Use the local copy instead.

        Args:
            session_id (SessionId): The session.

            name (str): The name of the blob. Blob names cannot start with a
                '.'.

        Yields:
            IO[bytes]: The file handle to write to.
        """
        with self._session_blob.open_blob_write(session_id, name) as b_out:
            yield b_out

    @contextmanager
    def open_blob_read(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        """
        Opens a blob for reading. This method is used for synchronization and
        should not be normally used. Use the local copy instead.

        Args:
            session_id (SessionId): The session.

            name (str): The name of the blob. Blob names cannot start with a
                '.'.

        Yields:
            IO[bytes]: The file handle to read from.
        """
        with self._session_blob.open_blob_read(session_id, name) as b_in:
            yield b_in

    def blob_hash(
            self,
            session_id: SessionId,
            names: Iterable[str]) -> dict[str, str]:
        """
        Computes the hash of remote blobs. See `get_file_hash` for the
        hashing method.

        Args:
            session_id (SessionId): The session.

            names (Iterable[str]): The name of the blobs.

        Returns:
            dict[str, str]: A mapping of names to blob hashes.
        """
        return self._session_blob.blob_hash(session_id, names)

    def blob_list(self, session_id: SessionId) -> list[str]:
        """
        Lists all blobs for the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            list[str]: The list of blob names.
        """
        return self._session_blob.blob_list(session_id)

    def blob_remove(self, session_id: SessionId, names: list[str]) -> None:
        """
        Remove the given blobs.

        Args:
            session_id (SessionId): The session.

            names (list[str]): The name of the blobs.
        """
        self._session_blob.blob_remove(session_id, names)

    def local_folder(self, session_id: SessionId) -> str:
        """
        The local folder where the local copy of the blobs is stored.

        Args:
            session_id (SessionId): The session.

        Returns:
            str: The path.
        """
        return os.path.join(self._cache_path, *session_id.as_folder())

    def scratch_folder(self) -> str:
        """
        A folder for temporary files.

        Returns:
            str: The path.
        """
        return self._scratch_path

    def locals(self) -> Iterable[SessionId]:
        """
        Find all session ids that have local copies.

        Yields:
            SessionId: The sessions.
        """
        yield from SessionId.find_folder_ids(self._cache_path)

    def clear_local(self, session_id: SessionId) -> None:
        """
        Immediately empty the local copy of the session. If called outside of
        a synchronization block only local disk space is freed.

        Args:
            session_id (SessionId): The session.
        """
        path = self.local_folder(session_id)
        shutil.rmtree(path, ignore_errors=True)

    def get_info(self, session_id: SessionId) -> InfoObj:
        """
        Returns information about the local copy. This information can be used
        to determine whether a local copy is currently in active use and
        whether it is safe to free up the space.

        Args:
            session_id (SessionId): The session.

        Returns:
            InfoObj: The info.
        """
        path = self.local_folder(session_id)
        try:
            with open_reads(os.path.join(path, INFO_NAME)) as fin:
                obj = json.load(fin)
                return {
                    "time_in": maybe_parse_time_str(obj.get("time_in")),
                    "time_out": maybe_parse_time_str(obj.get("time_out")),
                }
        except FileNotFoundError:
            return {
                "time_in": None,
                "time_out": None,
            }

    def set_info(self, session_id: SessionId, info: InfoObj) -> None:
        """
        Updates the info of the local copy.

        Args:
            session_id (SessionId): The session.

            info (InfoObj): The new info.
        """
        path = self.local_folder(session_id)
        with open_writes(os.path.join(path, ".info")) as fout:
            print(json.dumps({
                "time_in": maybe_fmt_time(info.get("time_in")),
                "time_out": maybe_fmt_time(info.get("time_out")),
            }, sort_keys=True, indent=2), file=fout)

    def sync_in(self, session_id: SessionId) -> None:
        """
        Populates the local copy of the given session.

        Args:
            session_id (SessionId): The session.
        """
        path = self.local_folder(session_id)
        blobs: set[str] = set(self.blob_list(session_id))
        need_copy: set[str] = set(blobs)
        need_hash: list[str] = []
        for fname in get_files(
                path,
                exclude_prefix=["."],
                exclude_ext=[TMP_POSTFIX],
                recurse=True):
            full_path = os.path.join(path, fname)
            if fname not in blobs:
                remove_file(full_path, stop=path)
                continue
            need_hash.append(fname)
        if need_hash:
            hash_lookup = self.blob_hash(session_id, need_hash)
        else:
            hash_lookup = {}
        for fname in need_hash:
            full_path = os.path.join(path, fname)
            own_hash = get_file_hash(full_path)
            in_hash = hash_lookup[fname]
            if own_hash == in_hash:
                need_copy.discard(fname)
        new_blobs = need_copy.difference(need_hash)
        if new_blobs:
            for fname, blob_hash in self.blob_hash(
                    session_id, new_blobs).items():
                hash_lookup[fname] = blob_hash
        for fname in need_copy:
            full_path = os.path.join(path, fname)
            ensure_folder(os.path.dirname(full_path))
            with open_writeb(full_path) as fout:
                with self.open_blob_read(session_id, fname) as fin:
                    shutil.copyfileobj(fin, fout)  # type: ignore
            assert get_file_hash(full_path) == hash_lookup[fname]
        self.set_info(session_id, {
            "time_in": now(),
            "time_out": None,
        })

    def sync_out(self, session_id: SessionId) -> None:
        """
        Synchronizes the local copy of the given session. This ensures that
        remote blobs are exactly the same as the local copy. If blobs do not
        exist locally anymore they are deleted remotely as well.

        Args:
            session_id (SessionId): The session.
        """
        # FIXME: allow folders
        path = self.local_folder(session_id)
        local: set[str] = set(get_files(
            path,
            exclude_prefix=["."],
            exclude_ext=[TMP_POSTFIX],
            recurse=True))
        need_copy: set[str] = set(local)
        need_hash: list[str] = []
        need_remove: list[str] = []
        for fname in self.blob_list(session_id):
            if fname not in local:
                need_remove.append(fname)
                continue
            need_hash.append(fname)
        if need_remove:
            self.blob_remove(session_id, need_remove)
        if need_hash:
            hash_lookup = self.blob_hash(session_id, need_hash)
        else:
            hash_lookup = {}
        out_hash_lookup: dict[str, str] = {}
        for fname in need_hash:
            full_path = os.path.join(path, fname)
            other_hash = hash_lookup[fname]
            out_hash = get_file_hash(full_path)
            if other_hash == out_hash:
                need_copy.discard(fname)
            else:
                out_hash_lookup[fname] = out_hash
        if need_copy:
            for fname in need_copy:
                full_path = os.path.join(path, fname)
                with self.open_blob_write(session_id, fname) as fout:
                    with open_readb(full_path) as fin:
                        shutil.copyfileobj(fin, fout)  # type: ignore
                if fname not in out_hash_lookup:
                    out_hash_lookup[fname] = get_file_hash(full_path)
            hash_final = self.blob_hash(session_id, need_copy)
            for fname in need_copy:
                assert out_hash_lookup[fname] == hash_final[fname]

        info = self.get_info(session_id)
        self.set_info(
            session_id,
            {
                "time_in": info.get("time_in"),
                "time_out": now(),
            })

    def copy_session_blobs(self, from_id: SessionId, to_id: SessionId) -> None:
        """
        Copies a session over from a different session. All values and blobs
        are copied over.

        Args:
            from_id (SessionId): The source session.

            to_id (SessionId): The destination session.
        """
        for key in self.get_keys(from_id):
            ix = 0
            while True:
                val = self.get_value(from_id, key, ix)
                if val is None:
                    break
                self.set_value(to_id, key, ix, val)
                ix += 1
        self.clear_local(to_id)
        path_to = self.local_folder(to_id)
        blobs = self.blob_list(from_id)
        hash_lookup = self.blob_hash(from_id, blobs)
        for fname in blobs:
            full_path = os.path.join(path_to, fname)
            ensure_folder(os.path.dirname(full_path))
            with open_writeb(full_path) as fout:
                with self.open_blob_read(from_id, fname) as fin:
                    shutil.copyfileobj(fin, fout)  # type: ignore
            assert get_file_hash(full_path) == hash_lookup[fname]
        self.set_info(to_id, {
            "time_in": now(),
            "time_out": None,
        })
        self.sync_out(to_id)

    def remove(self, session_id: SessionId) -> None:
        """
        Removes everything associated with the given session. Note, this does
        not guarantee that all local copies are removed as well.

        Args:
            session_id (SessionId): The session.
        """
        self._session_key_value.remove(session_id)
        self._session_blob.remove(session_id)
        self._session_user.remove(session_id)
        self.clear_local(session_id)


class SessionUser(Module):
    """Session handler to connect users with sessions."""
    def register_session(self, user_id: UserId, session_id: SessionId) -> None:
        """
        Registers the session so the user and session can be associated.

        Args:
            user_id (UserId): The user.
            session_id (SessionId): The session.
        """
        raise NotImplementedError()

    def get_sessions(
            self,
            sessions: SessionStore,
            user_id: UserId) -> Iterable[Session]:
        """
        Retrieves all sessions of the given user.

        Args:
            sessions (SessionHandler): The session handler.

            user_id (UserId): The user.

        Returns:
            Iterable[Session]: The sessions for the user.
        """
        raise NotImplementedError()

    def get_user(self, session_id: SessionId) -> UserId | None:
        """
        Retrieves the user associated with the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            UserId | None: The user or None if the session is not associated
                with any user (e.g., when it was deleted).
        """
        raise NotImplementedError()

    def remove(self, session_id: SessionId) -> None:
        """
        Removes everything associated with the given session. Note, this does
        not guarantee that all local copies are removed as well.

        Args:
            session_id (SessionId): The session.
        """
        raise NotImplementedError()


class SessionBlob(Module):
    """Session handler for providing a blob store for sessions."""
    @contextmanager
    def open_blob_write(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        """
        Opens a blob for writing. This method is used for synchronization and
        should not be normally used. Use the local copy instead.

        Args:
            session_id (SessionId): The session.

            name (str): The name of the blob. Blob names cannot start with a
                '.'.

        Yields:
            IO[bytes]: The file handle to write to.
        """
        raise NotImplementedError()

    @contextmanager
    def open_blob_read(
            self, session_id: SessionId, name: str) -> Iterator[IO[bytes]]:
        """
        Opens a blob for reading. This method is used for synchronization and
        should not be normally used. Use the local copy instead.

        Args:
            session_id (SessionId): The session.

            name (str): The name of the blob. Blob names cannot start with a
                '.'.

        Yields:
            IO[bytes]: The file handle to read from.
        """
        raise NotImplementedError()

    def blob_hash(
            self,
            session_id: SessionId,
            names: Iterable[str]) -> dict[str, str]:
        """
        Computes the hash of remote blobs. See `get_file_hash` for the
        hashing method.

        Args:
            session_id (SessionId): The session.

            names (Iterable[str]): The name of the blobs.

        Returns:
            dict[str, str]: A mapping of names to blob hashes.
        """
        raise NotImplementedError()

    def blob_list(self, session_id: SessionId) -> list[str]:
        """
        Lists all blobs for the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            list[str]: The list of blob names.
        """
        raise NotImplementedError()

    def blob_remove(self, session_id: SessionId, names: list[str]) -> None:
        """
        Remove the given blobs.

        Args:
            session_id (SessionId): The session.

            names (list[str]): The name of the blobs.
        """
        raise NotImplementedError()

    def remove(self, session_id: SessionId) -> None:
        """
        Removes everything associated with the given session. Note, this does
        not guarantee that all local copies are removed as well.

        Args:
            session_id (SessionId): The session.
        """
        raise NotImplementedError()


class SessionKeyValue(Module):
    """Session handler for providing a key value store for sessions."""
    def set_value(
            self,
            session_id: SessionId,
            key: str,
            index: int,
            value: str) -> None:
        """
        Sets the array value at the given index. If the index is equal to the
        length of the array a new value is appended. Negative index values
        address the array from the back. If the index is outside the array and
        it is not the positive length of the array an `IndexError` is raised.

        Raises:
            IndexError: If the index is outside the array and it is not the
                positive length of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            index (int): The index.
            value (str): The value to set.
        """
        raise NotImplementedError()

    def push_value(self, session_id: SessionId, key: str, value: str) -> None:
        """
        Appends a value to the end of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            value (str): The value.
        """
        raise NotImplementedError()

    def pop_value(self, session_id: SessionId, key: str) -> str | None:
        """
        Removes and returns the last value of the array.

        Args:
            session_id (SessionId): The session.
            key (str): The key.

        Returns:
            str | None: The value at the end of the array or None if the array
                was empty.
        """
        raise NotImplementedError()

    def get_value(
            self, session_id: SessionId, key: str, index: int) -> str | None:
        """
        Gets the value at the given index of the array. A negative index gets
        the value from the back of the array. Out of bounds access returns
        None.

        Args:
            session_id (SessionId): The session.
            key (str): The key.
            index (int): The index.

        Returns:
            str | None: The value or None if the index was out of bounds.
        """
        raise NotImplementedError()

    def get_length(self, session_id: SessionId, key: str) -> int:
        """
        Gets the length of the given array. If the key does not exist 0 is
        returned.

        Args:
            session_id (SessionId): The session.
            key (str): The key.

        Returns:
            int: The length of the array.
        """
        raise NotImplementedError()

    def get_keys(self, session_id: SessionId) -> list[str]:
        """
        Returns all keys of the given session.

        Args:
            session_id (SessionId): The session.

        Returns:
            list[str]: The list of all keys.
        """
        raise NotImplementedError()

    def notify_signal(self, session_id: SessionId, key: str) -> None:
        """
        Emits a signal on the given key. Note, that signal keys are independent
        of value keys.

        Args:
            session_id (SessionId): The session.
            key (str): The signal key.
        """
        raise NotImplementedError()

    def wait_for_signal(
            self,
            session_id: SessionId,
            key: str,
            condition: Callable[[], T],
            *,
            timeout: float) -> T | None:
        """
        Waits for a signal on the given key and returns if the condition is
        met. If no signal arrives or the condition is not met within the
        timeout False is returned. Note, that signal keys are independent
        of value keys.

        Args:
            session_id (SessionId): The session.

            key (str): The signal key

            condition (Callable[[], T]): If the condition can be converted to
                True, the method returns successfully with its result.

            timeout (float): The timeout in seconds.

        Returns:
            T | None: The result of the condition if it could be evaluated to
                True. Otherwise, if the timeout occurred, None is returned.
        """
        raise NotImplementedError()

    def remove(self, session_id: SessionId) -> None:
        """
        Removes everything associated with the given session. Note, this does
        not guarantee that all local copies are removed as well.

        Args:
            session_id (SessionId): The session.
        """
        raise NotImplementedError()
