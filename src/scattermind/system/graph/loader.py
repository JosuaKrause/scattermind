
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
