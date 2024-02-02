# Scattermind distributes computation of machine learning models.
# Copyright (C) 2024 Josua Krause
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Loads the scattermind API."""
from scattermind.api.api import ScattermindAPI
from scattermind.system.config.loader import ConfigJSON, load_as_api


def load_api(config_obj: ConfigJSON) -> ScattermindAPI:
    """
    Load a scattermind API from a JSON.

    Args:
        config_obj (ConfigJSON): The configuration JSON.

    Returns:
        Config: The scattermind API.
    """
    return load_as_api(config_obj)
