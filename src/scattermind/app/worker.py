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
"""A scattermind worker process."""
import json
from typing import cast

from scattermind.system.base import ExecutorId
from scattermind.system.config.loader import ConfigJSON, load_config
from scattermind.system.graph.graphdef import FullGraphDefJSON


def worker_start(*, config_file: str, graph_def_file: str) -> None:
    """
    Load configuration, graph, and start execution.

    Args:
        config_file (str): The configuration file.
        graph_def_file (str): The graph definition file.
    """
    with open(config_file, "rb") as fin:
        config_obj = cast(ConfigJSON, json.load(fin))
    config = load_config(ExecutorId.create, config_obj)
    with open(graph_def_file, "rb") as fin:
        graph_def_obj = cast(FullGraphDefJSON, json.load(fin))
    config.load_graph(graph_def_obj)
    config.run()
