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
"""Loads a node."""
from typing import TYPE_CHECKING

from scattermind.system.base import NodeId
from scattermind.system.graph.node import INTERNAL_NODE_PREFIX, Node
from scattermind.system.plugins import load_plugin


if TYPE_CHECKING:
    from scattermind.system.graph.graph import Graph


# FIXME determine the valid nodes from the nodes folder and create a fitting
# error message
def load_node(
        graph: 'Graph',
        kind: str,
        node_id: NodeId) -> 'Node':
    """
    Load a node for the given graph.

    Args:
        graph (Graph): The graph.
        kind (str): The kind of the node. This can be built-in node names or
            fully qualified python module names to load a node via plugin.
        node_id (NodeId): The node id to assign to the ndoe.

    Raises:
        ModuleNotFoundError: If provided a python module name that cannot be
            resolved.
        ValueError: If the built-in name is not known.

    Returns:
        Node: The loaded node.
    """
    kind_name = kind
    if "." not in kind:
        kind = f"{INTERNAL_NODE_PREFIX}.{kind}"
    try:
        node_cls = load_plugin(Node, kind)
    except ModuleNotFoundError as exc:
        if kind == kind_name:
            raise exc
        raise ValueError(f"unknown node {kind_name}") from exc
    return node_cls(kind_name, graph, node_id)
