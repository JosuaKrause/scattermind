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
from typing import Literal, TypedDict

from scattermind.system.plugins import load_plugin
from scattermind.system.session.session import SessionStore


LocalSessionStoreModule = TypedDict('LocalSessionStoreModule', {
    "name": Literal["ram"],
    "path": str,
})


SessionStoreModule = LocalSessionStoreModule


def load_session_store(module: SessionStoreModule) -> SessionStore:
    # pylint: disable=import-outside-toplevel
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(SessionStore, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "ram":
        pass  # FIXME TODO
    raise ValueError(f"unknown readonly access: {module['name']}")
