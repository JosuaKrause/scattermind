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
"""A JSON specification of a graph."""
from typing import TypedDict

from typing_extensions import NotRequired

from scattermind.system.base import GraphId, NodeId, QueueId
from scattermind.system.graph.args import NodeArg, NodeArguments
from scattermind.system.graph.graph import Graph
from scattermind.system.info import DataFormat, DataFormatJSON
from scattermind.system.names import (
    GName,
    GNamespace,
    NName,
    QName,
    QualifiedGraphName,
    QualifiedName,
)
from scattermind.system.queue.queue import QueuePool


OutDefJSON = dict[str, str | None]
"""
A dictionary to map all output queue names of a node to a node name and
input queue name. The node and input can be `None` to indicate a mapping to
the graph output. The node and input pair can also be just a queue id.
"""

ValueMapJSON = dict[str, str]
"""
A dictionary to map all work names to input qualified names.
A work name is the name of a value as specified by a node.
A qualified name is `f"{node_name}:{value_name}"` or `f":{value_name}"`
for inputs.
"""

NodeDefJSON = TypedDict('NodeDefJSON', {
    "node_id": NotRequired[str],
    "name": str,
    "kind": str,
    "args": NotRequired[NodeArguments],
    "qin": NotRequired[str],
    "outs": NotRequired[OutDefJSON],
    "vmap": NotRequired[ValueMapJSON],
})
"""
Node definition. `node_id` is optional. `name` is a readable name. `kind` is
the node kind. `args` are node arguments. `qin` is the optional input queue id.
`outs` defines where the task goes next for a given output. `vmap` maps the
inputs of the node to locations in the stack frame.
"""

# FIXME implement blocks
# - blocks are computed with separate local everything
# - can only be executed from call
# - nodes/queues are kept in a separate location (different queue_pool)
# - all nodes from a block are loaded at once (happens in the calling node)
# - can only call other blocks
# - optimizations can be turn chain of nodes into a block and call
# - find mutually recursive calls and turn them into a block together
# - don't separate out data -- can be kept in memory to point to the same bulk
# tensor still
GraphDefJSON = TypedDict('GraphDefJSON', {
    "graph_id": NotRequired[str],
    "name": str,
    "description": NotRequired[str],
    "input": str,
    "input_format": DataFormatJSON,
    "output_format": DataFormatJSON,
    "vmap": ValueMapJSON,
    "nodes": list[NodeDefJSON],
    "is_block": NotRequired[bool],
})
"""
Graph definition. `name` and `description` are for information purposes only.
`graph_id` is the optional fixed graph id. It is inferred from the name if not
specified. `input` specifies the node where the initial input is pushed to.
`input_format` specifies the expected format of the graph input.
`output_format` specified the expected format of the graph output. `vmap` maps
the outputs of the graph to locations in the stack frame. `nodes` is a list of
all nodes of the graph. `is_block` specifies that the whole graph should be
computed in one go instead pushing tasks to queues and picking them up for
computation. If a graph is a block, only the input queue of the graph is used.
"""

FullGraphDefJSON = TypedDict('FullGraphDefJSON', {
    "graphs": list[GraphDefJSON],
    "entry": str,
    "ns": NotRequired[str],
})


def graph_to_json(graph: Graph, queue_pool: QueuePool) -> FullGraphDefJSON:
    """
    Creates a JSON serializable representation of a graph.

    Args:
        graph (Graph): The graph.
        queue_pool (QueuePool): The associated queue pool.

    Returns:
        FullGraphDefJSON: The JSON serializable object.
    """
    ns = graph.get_namespace()
    graphs = []
    for graph_id in queue_pool.get_graphs():
        nodes: list[NodeDefJSON] = [
            {
                "kind": node.get_kind(),
                "node_id": node.get_id().to_parseable(),
                "name": node.get_name().get(),
                "args": NodeArg.to_node_arguments(node.get_node_arguments()),
                "qin": node.get_input_queue().to_parseable(),
                "outs": {
                    output_name:
                        node.get_output_queue(
                            QName(output_name)).to_parseable()
                    for output_name in node.get_outputs()
                },
                "vmap": {
                    wname: qual.to_parseable()
                    for wname, qual in graph.get_value_map(
                        node.get_id()).items()
                },
            }
            for node in graph.traverse_graph(queue_pool, graph_id)
        ]
        input_node = queue_pool.get_input_node(graph_id)
        vmap = queue_pool.get_output_value_map(graph_id)
        gdef: GraphDefJSON = {
            "graph_id": graph_id.to_parseable(),
            "name": queue_pool.get_graph_name(graph_id).get_name().get(),
            "description": queue_pool.get_graph_description(graph_id),
            "input": input_node.get_name().get(),
            "input_format":
                queue_pool.get_input_format(graph_id).data_format_to_json(),
            "output_format":
                queue_pool.get_output_format(graph_id).data_format_to_json(),
            "nodes": nodes,
            "vmap": {
                wname: qual.to_parseable()
                for wname, qual in vmap.items()
            },
        }
        graphs.append(gdef)
    return {
        "graphs": graphs,
        "entry": queue_pool.get_graph_name(
            queue_pool.get_entry_graph(ns)).get_name().get(),
        "ns": ns.get(),
    }


def json_to_graph(queue_pool: QueuePool, def_obj: FullGraphDefJSON) -> Graph:
    """
    Parses a JSON graph definition and converts it into a graph object.

    Args:
        queue_pool (QueuePool): The queue pool.
        def_obj (FullGraphDefJSON): The JSON graph definition.

    Raises:
        ValueError: If the graph could not be loaded.

    Returns:
        Graph: The graph object.
    """
    ns_val = def_obj.get("ns")
    if ns_val is None:
        ns = GNamespace(def_obj["entry"])
    else:
        ns = GNamespace(ns_val)
    graph = Graph(ns)
    for gobj in def_obj["graphs"]:
        gname = QualifiedGraphName(ns, GName(gobj["name"]))
        graph_id_str = gobj.get("graph_id")
        if graph_id_str is not None:
            graph_id = GraphId.parse(graph_id_str)
        else:
            graph_id = GraphId.create(gname)
        queue_pool.add_graph(graph_id, gname, gobj.get("description", ""))
        node_ids: dict[NName, NodeId] = {}
        for node_obj in gobj["nodes"]:
            node_name = NName(node_obj["name"])
            if node_name in node_ids:
                raise ValueError(f"duplicate node name: {node_name}")
            node_kind = node_obj["kind"]
            node_id_str = node_obj.get("node_id")
            if node_id_str is None:
                node_id = NodeId.create(gname, node_name)
            else:
                node_id = NodeId.parse(node_id_str)
            node_ids[node_name] = node_id
            node_args = node_obj.get("args", {})
            fixed_qin_str = node_obj.get("qin")
            if fixed_qin_str is not None:
                fixed_qin = QueueId.parse(fixed_qin_str)
            else:
                fixed_qin = None
            vmap = {
                wname: QualifiedName.parse(fname)
                for wname, fname in node_obj.get("vmap", {}).items()
            }
            graph.add_node(
                queue_pool,
                graph_id,
                kind=node_kind,
                name=node_name,
                node_id=node_id,
                args=node_args,
                fixed_input_queue_id=fixed_qin,
                vmap=vmap)
        queue_pool.set_input_format(
            graph_id,
            DataFormat.data_format_from_json(gobj["input_format"]))
        queue_pool.set_output_format(
            graph_id,
            DataFormat.data_format_from_json(gobj["output_format"]))
        for node_obj in gobj["nodes"]:
            out_node_id = node_ids[NName(node_obj["name"])]
            for out_name, in_def in node_obj.get("outs", {}).items():
                if in_def is None:
                    out_qid = QueueId.get_output_queue()
                else:
                    try:
                        out_qid = QueueId.parse(in_def)
                    except ValueError:
                        in_def_node = graph.get_node(node_ids[NName(in_def)])
                        out_qid = in_def_node.get_input_queue()
                graph.add_edge(
                    output_node_id=out_node_id,
                    output_queue=QName(out_name),
                    input_queue_id=out_qid)
        input_node = graph.get_node(node_ids[NName(gobj["input"])])
        queue_pool.set_input_node(graph_id, input_node)
        queue_pool.set_output_value_map(
            graph_id,
            {
                wname: QualifiedName.parse(fname)
                for wname, fname in gobj["vmap"].items()
            })
    try:
        entry_id = GraphId.parse(def_obj["entry"])
    except ValueError:
        entry_id = queue_pool.get_graph_id(
            QualifiedGraphName(ns, GName(def_obj["entry"])))
    queue_pool.set_entry_graph(ns, entry_id)
    return graph
