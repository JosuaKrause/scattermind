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
"""A scattermind worker process."""
import json
import os
from typing import cast

from scattermind.system.base import ExecutorId
from scattermind.system.config.config import Config
from scattermind.system.config.loader import ConfigJSON, load_config
from scattermind.system.graph.graphdef import FullGraphDefJSON
from scattermind.system.torch_util import set_system_device


def worker_start(
        *, config_file: str, graph_def: str, device: str | None) -> None:
    """
    Load configuration, graph, and start execution.

    Args:
        config_file (str): The configuration file.
        graph_def (str): The graph definition file or folder containing
            graph definition files.
        device (str): Overrides the system device if set.
    """
    if device is not None:
        set_system_device(device)

    with open(config_file, "rb") as fin:
        config_obj = cast(ConfigJSON, json.load(fin))
    config: Config = load_config(ExecutorId.create, config_obj)

    def load_graph(graph_file: str) -> None:
        with open(graph_file, "rb") as fin:
            graph_def_obj = cast(FullGraphDefJSON, json.load(fin))
        config.load_graph(graph_def_obj)

    if os.path.isdir(graph_def):
        for name in os.listdir(graph_def):
            if not name.endswith(".json"):
                continue
            fname = os.path.join(graph_def, name)
            if not os.path.isfile(fname):
                continue
            load_graph(fname)
    else:
        load_graph(graph_def)
    config.run()
