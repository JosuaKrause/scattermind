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
from scattermind.system.session.session import SessionStore
from scattermind.system.util import to_bool


RedisSessionStoreModule = TypedDict('RedisSessionStoreModule', {
    "name": Literal["redis"],
    "cfg": RedisConfig,
    "disk_path": str,
    "cache_path": str,
    "is_shared": bool,
})


SessionStoreModule = RedisSessionStoreModule


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
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(SessionStore, f"{kwargs.pop('name')}")
        cache_path = f"{kwargs.pop('cache_path')}"
        is_shared = to_bool(kwargs.pop('is_shared'))
        return plugin(cache_path=cache_path, is_shared=is_shared, **kwargs)
    if module["name"] == "redis":
        from scattermind.system.session.redis import RedisSessionStore
        return RedisSessionStore(
            module["cfg"],
            disk_path=module["disk_path"],
            cache_path=module["cache_path"],
            is_shared=module["is_shared"])
    raise ValueError(f"unknown session store: {module['name']}")
