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
"""Loads the scattermind API."""
from typing import TypeAlias

from scattermind.api.api import ScattermindAPI
from scattermind.system.config.loader import ConfigJSON, load_as_api


VersionInfo: TypeAlias = tuple[str, str, str]


def get_version_info(version_file: str | None) -> VersionInfo | None:
    """
    Reads a version file.

    Args:
        version_file (str): The version file.

    Returns:
        VersionTuple: The version, hash, and deploy date.
    """
    if version_file is None:
        return None
    with open(version_file, "r", encoding="utf-8") as fin:
        version_name = fin.readline().strip()
        version_hash = fin.readline().strip()
        version_date = fin.readline().strip()
    return version_name, version_hash, version_date


def load_api(config_obj: ConfigJSON) -> ScattermindAPI:
    """
    Load a scattermind API from a JSON.

    Args:
        config_obj (ConfigJSON): The configuration JSON.

    Returns:
        Config: The scattermind API.
    """
    return load_as_api(config_obj)
