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
"""Create a session store."""
from typing import Literal, overload, TypedDict

from redipy import RedisConfig

from scattermind.system.plugins import load_plugin
from scattermind.system.session.session import (
    SessionBlob,
    SessionKeyValue,
    SessionStore,
    SessionUser,
)
from scattermind.system.util import to_bool


RedisSessionUserModule = TypedDict('RedisSessionUserModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})


SessionUserModule = RedisSessionUserModule


RedisSessionKeyValueModule = TypedDict('RedisSessionKeyValueModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
})


SessionKeyValueModule = RedisSessionKeyValueModule


DiskSessionBlobModule = TypedDict('DiskSessionBlobModule', {
    "name": Literal["disk"],
    "cfg": RedisConfig,
    "disk_path": str,
})


SessionBlobModule = DiskSessionBlobModule


SessionStoreModule = TypedDict('SessionStoreModule', {
    "user": SessionUserModule,
    "key_value": SessionKeyValueModule,
    "blob": SessionBlobModule,
    "cache_path": str,
    "is_shared": bool,
})


def load_session_user(module: SessionUserModule) -> SessionUser:
    """
    Loads the session user handler.

    Args:
        module (SessionUserModule): The module configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        SessionUser: The session user handler.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(SessionUser, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "redis":
        from scattermind.system.session.user_redis import RedisSessionUser
        return RedisSessionUser(module["cfg"])
    raise ValueError(f"unknown session user handler: {module['name']}")


def load_session_key_value(module: SessionKeyValueModule) -> SessionKeyValue:
    """
    Loads the session key value handler.

    Args:
        module (SessionKeyValueModule): The module configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        SessionKeyValue: The session key value handler.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(SessionKeyValue, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "redis":
        from scattermind.system.session.kv_redis import RedisSessionKeyValue
        return RedisSessionKeyValue(module["cfg"])
    raise ValueError(f"unknown session user handler: {module['name']}")


def load_session_blob(module: SessionBlobModule) -> SessionBlob:
    """
    Loads the session blob handler.

    Args:
        module (SessionBlobModule): The module configuration.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        SessionBlob: The session blob handler.
    """
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(SessionBlob, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "disk":
        from scattermind.system.session.blob_disk import DiskSessionBlob
        return DiskSessionBlob(module["cfg"], disk_path=module["disk_path"])
    raise ValueError(f"unknown session user handler: {module['name']}")


@overload
def load_session_store(
        module: SessionStoreModule) -> SessionStore:
    ...


@overload
def load_session_store(module: None) -> None:
    ...


def load_session_store(
        module: SessionStoreModule | None) -> SessionStore | None:
    """
    Loads the session store for the given configuration.

    Args:
        module (SessionStoreModule | None): The session store configuration or
            None if sessions are deactivated.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        SessionStore | None: The session store or None.
    """
    # pylint: disable=import-outside-toplevel
    if module is None:
        return None
    session_user = load_session_user(module["user"])
    session_blob = load_session_blob(module["blob"])
    session_kv = load_session_key_value(module["key_value"])
    return SessionStore(
        session_user,
        session_blob,
        session_kv,
        cache_path=module["cache_path"],
        is_shared=to_bool(module["is_shared"]))
