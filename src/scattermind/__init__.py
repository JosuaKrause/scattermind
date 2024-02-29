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


from typing import Any


PACKAGE_VERSION: str | None = None


def _get_version() -> str:
    # pylint: disable=import-outside-toplevel
    global PACKAGE_VERSION  # pylint: disable=global-statement

    if PACKAGE_VERSION is None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            PACKAGE_VERSION = version("scattermind")
        except PackageNotFoundError:
            try:
                import os
                import tomllib

                pyproject_fname = os.path.join(
                    os.path.dirname(__file__), "../../pyproject.toml")
                if (os.path.exists(pyproject_fname)
                        and os.path.isfile(pyproject_fname)):
                    with open(pyproject_fname, "rb") as fin:
                        pyproject = tomllib.load(fin)
                    if pyproject["project"]["name"] == "scattermind":
                        PACKAGE_VERSION = f"{pyproject['project']['version']}*"
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if PACKAGE_VERSION is None:
            PACKAGE_VERSION = "unknown"
    return PACKAGE_VERSION


def __getattr__(name: str) -> Any:
    if name in ("version", "__version__"):
        return _get_version()
    raise AttributeError(f"No attribute {name} in module {__name__}.")
