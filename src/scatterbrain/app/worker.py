import json
from typing import cast

from scatterbrain.system.base import ExecutorId
from scatterbrain.system.config.loader import ConfigJSON, load_config
from scatterbrain.system.graph.graphdef import FullGraphDefJSON


def worker_start(*, config_file: str, graph_def_file: str) -> None:
    with open(config_file, "rb") as fin:
        config_obj = cast(ConfigJSON, json.load(fin))
    config = load_config(ExecutorId.create, config_obj)
    with open(graph_def_file, "rb") as fin:
        graph_def_obj = cast(FullGraphDefJSON, json.load(fin))
    config.load_graph(graph_def_obj)
    config.run()
