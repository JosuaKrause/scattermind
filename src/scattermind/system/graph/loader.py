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
