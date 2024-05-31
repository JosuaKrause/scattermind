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
"""A collection of useful IO operations."""
import contextlib
import errno
import io
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import AnyStr, IO, overload


MAIN_LOCK = threading.RLock()
"""The main IO lock."""
STALE_FILE_RETRIES: list[float] = [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 5]
"""Wait times for stale file access."""
TMP_POSTFIX = ".~tmp"
"""Postfix for temporary files."""


RETRY_ERRORS: set[int] = {errno.EAGAIN, errno.EBUSY, errno.EROFS}


def when_ready(fun: Callable[[], None]) -> None:
    """
    Repeats an IO operation for a busy or slow disk device (e.g., NFS or
    disk access on docker start up) until it succeeds. The operation needs to
    be idempotent.

    Args:
        fun (Callable[[], None]): The IO operation to perform.
    """
    with MAIN_LOCK:
        counter = 0
        while True:
            try:
                fun()
                return
            except OSError as ose:
                if counter < 120 and ose.errno in RETRY_ERRORS:
                    time.sleep(1.0)
                    counter += 1
                    continue
                raise ose


def fastrename(src: str, dst: str) -> None:
    """
    Renames a file or folder.

    Args:
        src (str): The source file or folder.
        dst (str): The destination file or folder.
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if src == dst:
        raise ValueError(f"{src} == {dst}")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} does not exist!")

    def do_rename() -> None:
        # ensure_folder(os.path.dirname(dst))  # FIXME: is this necessary?
        os.rename(src, dst)

    try:
        when_ready(do_rename)
        if not src.endswith(TMP_POSTFIX):
            print(f"move {src} to {dst}")
    except OSError:
        for file_name in listdir(src):
            try:
                shutil.move(os.path.join(src, file_name), dst)
            except shutil.Error as err:
                dest_file = os.path.join(dst, file_name)
                err_msg = f"{err}".lower()
                if "destination path" in err_msg and \
                        "already exists" in err_msg:
                    raise err
                remove_file(dest_file)
                shutil.move(os.path.join(src, file_name), dst)


def copy_file(from_file: str, to_file: str) -> None:
    """
    Fully copy a file to a new location.

    Args:
        from_file (str): The source file.
        to_file (str): The destination.
    """
    shutil.copy(from_file, to_file)


def normalize_folder(folder: str) -> str:
    """
    Normalizes a folder path and ensures it exists.

    Args:
        folder (str): The folder.

    Returns:
        str: The normalized path.
    """
    res = os.path.abspath(folder)
    when_ready(lambda: os.makedirs(res, mode=0o777, exist_ok=True))
    if not os.path.isdir(res):
        raise ValueError(f"{folder} must be a folder")
    return res


def normalize_file(fname: str) -> str:
    """
    Normalizes a file path and ensures its folder exists.

    Args:
        fname (str): The file.

    Returns:
        str: The normalized path.
    """
    res = os.path.abspath(fname)
    normalize_folder(os.path.dirname(res))
    return res


def is_empty_file(fin: IO[AnyStr]) -> bool:
    """
    Checks whether a file is empty from the current cursor position.

    Args:
        fin (IO[AnyStr]): The file handle.

    Returns:
        bool: True if there is no more content to read from the file.
    """
    pos = fin.seek(0, io.SEEK_CUR)
    size = fin.seek(0, io.SEEK_END) - pos
    fin.seek(pos, io.SEEK_SET)
    return size <= 0


@overload
def ensure_folder(folder: str) -> str:
    ...


@overload
def ensure_folder(folder: None) -> None:
    ...


def ensure_folder(folder: str | None) -> str | None:
    """
    Ensures that a folder exist. Input can be `None` in which case the function
    is a no-op.

    Args:
        folder (str | None): The folder.

    Returns:
        str | None: The folder.
    """
    if folder is not None and not os.path.exists(folder):
        a_folder: str = folder
        when_ready(lambda: os.makedirs(a_folder, mode=0o777, exist_ok=True))
    return folder


def get_tmp(basefile: str) -> str:
    """
    Returns the path component of the given file. Also, ensures that the folder
    exists.

    Args:
        basefile (str): The file.

    Returns:
        str: The folder.
    """
    return ensure_folder(os.path.dirname(basefile))


@contextlib.contextmanager
def named_write(filename: str) -> Iterator[str]:
    """
    Provides a secure way to write to the given file that is robust against
    partial writes. The actual write is performed against a temporary file and
    only if the write completes successfully the temporary file is renamed to
    the correct path.

    Args:
        filename (str): The file destination.

    Yields:
        str: The name of the temporary file.
    """
    filename = normalize_file(filename)

    tname = None
    writeback = False
    try:
        tfd, tname = tempfile.mkstemp(
            dir=get_tmp(filename),
            suffix=TMP_POSTFIX)
        os.close(tfd)
        yield tname
        writeback = True
    finally:
        if tname is not None:
            if writeback:
                fastrename(tname, filename)
            else:
                remove_file(tname)


def remove_file(fname: str, *, stop: str | None = None) -> None:
    """
    Removes the given file.

    Args:
        fname (str): The path.

        stop (str | None, optional): The first path allowed to be an empty
            folder. If None, no additional folders are removed.
    """
    if stop is not None and not fname.startswith(stop):
        raise ValueError(f"{stop=} must be a prefix of {fname=}")
    try:
        os.remove(fname)
        if stop is not None:
            try:
                while fname and fname.startswith(stop) and fname != stop:
                    fname = os.path.dirname(fname)
                    os.rmdir(fname)
            except OSError as oserr:
                if oserr.errno != 66:
                    raise oserr
    except FileNotFoundError:
        pass


def get_subfolders(path: str) -> list[str]:
    """
    Gets all subfolders of the given folder.

    Args:
        path (str): The folder.

    Returns:
        list[str]: A sorted list of all subfolder names.
    """
    try:
        return sorted(
            (fobj.name for fobj in os.scandir(path) if fobj.is_dir()))
    except FileNotFoundError:
        return []


def get_files(
        path: str,
        *,
        exclude_prefix: list[str] | None = None,
        exclude_ext: list[str] | None = None,
        recurse: bool = False,
        path_prefix: str | None = None) -> Iterable[str]:
    """
    Get all files of the given folder and extension.

    Args:
        path (str): The folder.

        exclude_prefix (list[str] | None, optional): A list of prefixes to
            exclude. Defaults to no exclusions.

        exclude_ext (list[str] | None, optional): A list of extensions to
            exclude. Defaults to no exclusions.

        recurse (bool, optional): Whether to include subfolders. Filenames are
            relative to the folder. Default is False.

        path_prefix (str | None, optional): A path prefix to include in the
            results.

    Yields:
        str: All filenames.
    """
    if exclude_prefix is None:
        exclude_prefix = []
    if exclude_ext is None:
        exclude_ext = []
    if path_prefix is None:
        path_prefix = ""
    try:
        for fobj in os.scandir(path):
            if any(fobj.name.endswith(ext) for ext in exclude_ext):
                continue
            if any(fobj.name.startswith(prefix) for prefix in exclude_prefix):
                continue
            cur_name = os.path.join(path_prefix, fobj.name)
            if fobj.is_file():
                yield cur_name
            elif recurse and fobj.is_dir():
                sub_path = os.path.join(path, fobj.name)
                yield from get_files(
                    sub_path,
                    exclude_prefix=exclude_prefix,
                    exclude_ext=exclude_ext,
                    recurse=recurse,
                    path_prefix=cur_name)
    except FileNotFoundError:
        yield from []


def get_folder(path: str, ext: str) -> Iterable[tuple[str, bool]]:
    """
    Get all files in the given folder and extension (for files only).

    Args:
        path (str): The folder.
        ext (str): The extension.

    Yields:
        tuple[str, bool]: A tuple of the name and True if it is a folder,
            otherwise False if it is a file.
    """
    try:
        for fobj in sorted(os.scandir(path), key=lambda fobj: fobj.name):
            if fobj.is_dir():
                yield fobj.name, True
            elif fobj.is_file() and fobj.name.endswith(ext):
                yield fobj.name, False
    except FileNotFoundError:
        yield from []


def listdir(path: str) -> list[str]:
    """
    Lists all filenames of the given folder path.

    Args:
        path (str): The folder.

    Returns:
        list[str]: A sorted list of all filenames in the folder. An empty list
            is returned if the path does not exist.
    """
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []


@contextlib.contextmanager
def open_reads(filename: str) -> Iterator[IO[str]]:
    """
    Open a file for reading in text mode. This function ensures that the
    file is ready even on slow disk drives (e.g., NFS or docker disk
    immediately after start up). The file must not be empty (o bytes long).

    Args:
        filename (str): The file.

    Yields:
        IO[str]: The file handle for reading.
    """
    ix = 0
    yielded = False
    sfr = STALE_FILE_RETRIES
    next_sleep: float | None = None
    while not yielded:
        if next_sleep is not None:
            time.sleep(next_sleep)
        with open(filename, "r", encoding="utf-8") as res:
            try:
                if is_empty_file(res) and ix < len(sfr):
                    next_sleep = sfr[ix]
                    ix += 1
                    continue
                yielded = True
                yield res
            except OSError as os_err:
                if yielded:
                    raise os_err
                if ix >= len(sfr) or os_err.errno != errno.ESTALE:
                    raise os_err
                next_sleep = sfr[ix]
                ix += 1


@contextlib.contextmanager
def open_readb(filename: str) -> Iterator[IO[bytes]]:
    """
    Open a file for reading in binary mode. This function ensures that the
    file is ready even on slow disk drives (e.g., NFS or docker disk
    immediately after start up). The file must not be empty (o bytes long).

    Args:
        filename (str): The file.

    Yields:
        IO[bytes]: The file handle for reading.
    """
    ix = 0
    yielded = False
    sfr = STALE_FILE_RETRIES
    next_sleep: float | None = None
    while not yielded:
        if next_sleep is not None:
            time.sleep(next_sleep)
        with open(filename, "rb") as res:
            try:
                if is_empty_file(res) and ix < len(sfr):
                    next_sleep = sfr[ix]
                    ix += 1
                    continue
                yielded = True
                yield res
            except OSError as os_err:
                if yielded:
                    raise os_err
                if ix >= len(sfr) or os_err.errno != errno.ESTALE:
                    raise os_err
                next_sleep = sfr[ix]
                ix += 1


@contextlib.contextmanager
def open_writes(
        filename: str,
        *,
        tmp_base: str | None = None,
        filename_fn: Callable[[str, str], str] | None = None,
        ) -> Iterator[IO[str]]:
    """
    Open a file for writing in text mode. This function makes its best effort
    to ensure the file is available even on slow disk drives (e.g., NFS). It
    also secures against incomplete writes by writing to a temporary file first
    and moving the file to the real location only after the initial write is
    complete.

    Args:
        filename (str): The filename.

        tmp_base (str | None, optional): If set the temporary file will be
            created in this folder. Otherwise the folder of the initial
            filename is used instead (this will cause the filename to be
            normalized and its folder to be created).

        filename_fn (Callable[[str, str], str] | None, optional): Adjusts the
            filename before writing the file back. The arguments to the
            function are the initial filename and the temporary filename.
            Defaults to the initial filename.

    Yields:
        IO[str]: The file handle.
    """
    if filename_fn is None or tmp_base is None:
        filename = normalize_file(filename)

    if tmp_base is None:
        tmp_base = get_tmp(filename)
    else:
        ensure_folder(tmp_base)

    def fname_id(fname: str, _tmp: str) -> str:
        return fname

    if filename_fn is None:
        filename_fn = fname_id

    tname = None
    tfd = None
    sfile: IO[str] | None = None
    writeback = False
    try:
        tfd, tname = tempfile.mkstemp(
            dir=tmp_base,
            suffix=TMP_POSTFIX)
        bfile = io.FileIO(tfd, "wb", closefd=True)
        sfile = io.TextIOWrapper(bfile, encoding="utf-8", line_buffering=True)
        yield sfile
        sfile.flush()
        os.fsync(tfd)
        writeback = True
    finally:
        if sfile is not None:
            sfile.close()  # closes the temporary file descriptor
        elif tfd is not None:
            os.close(tfd)  # closes the actual temporary file descriptor
        if tname is not None:
            if writeback:
                adj_fname = filename_fn(filename, tname)
                fastrename(tname, adj_fname)
            else:
                remove_file(tname)


@contextlib.contextmanager
def open_writeb(
        filename: str,
        *,
        tmp_base: str | None = None,
        filename_fn: Callable[[str, str], str] | None = None,
        ) -> Iterator[IO[bytes]]:
    """
    Open a file for writing in binary mode. This function makes its best effort
    to ensure the file is available even on slow disk drives (e.g., NFS). It
    also secures against incomplete writes by writing to a temporary file first
    and moving the file to the real location only after the initial write is
    complete.

    Args:
        filename (str): The filename.

        tmp_base (str | None, optional): If set the temporary file will be
            created in this folder. Otherwise the folder of the initial
            filename is used instead (this will cause the filename to be
            normalized and its folder to be created).

        filename_fn (Callable[[str, str], str] | None, optional): Adjusts the
            filename before writing the file back. The arguments to the
            function are the initial filename and the temporary filename.
            Defaults to the initial filename.

    Yields:
        IO[bytes]: The file handle.
    """
    if filename_fn is None or tmp_base is None:
        filename = normalize_file(filename)

    if tmp_base is None:
        tmp_base = get_tmp(filename)
    else:
        ensure_folder(tmp_base)

    def fname_id(fname: str, _tmp: str) -> str:
        return fname

    if filename_fn is None:
        filename_fn = fname_id

    tname = None
    tfd = None
    sfile: IO[bytes] | None = None
    writeback = False
    try:
        tfd, tname = tempfile.mkstemp(
            dir=tmp_base,
            suffix=TMP_POSTFIX)
        sfile = io.FileIO(tfd, "wb", closefd=True)
        yield sfile
        sfile.flush()
        os.fsync(tfd)
        writeback = True
    finally:
        if sfile is not None:
            sfile.close()  # closes the temporary file descriptor
        elif tfd is not None:
            os.close(tfd)  # closes the actual temporary file descriptor
        if tname is not None:
            if writeback:
                adj_fname = filename_fn(filename, tname)
                fastrename(tname, adj_fname)
            else:
                remove_file(tname)
