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
"""Scattermind is a decentralized and distributed horizontally scalable model
execution framework."""


def _get_version() -> str:
    # pylint: disable=import-outside-toplevel
    try:
        import os

        import tomllib

        with open(os.path.join(__file__, "../pyproject.toml"), "rb") as fin:
            pyproject = tomllib.load(fin)
        if pyproject["project"]["name"] == "scattermind":
            return pyproject["project"]["version"]
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    from importlib.metadata import version

    return version("scattermind")


__version__ = _get_version()  # pylint: disable=invalid-name
