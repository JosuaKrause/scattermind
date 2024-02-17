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
"""Provides node arguments."""
from typing import cast, Literal, overload

from scattermind.system.info import DataFormatJSON, DataInfo, DataInfoJSON
from scattermind.system.names import (
    GName,
    GNamespace,
    NAME_SEP,
    QualifiedGraphName,
)
from scattermind.system.readonly.access import DataAccess
from scattermind.system.torch_util import DTypeName, get_dtype_name
from scattermind.system.util import as_int_list, as_shape


ArgType = Literal[
    "str",
    "int",
    "float",
    "list_int",
    "shape",
    "ushape",
    "format",
    "info",
    "graph",
    "dtype",
    "data",
]
"""Available argument types as string names."""


ArgumentType = (
    str
    | int
    | float
    | list[int]
    | list[int | None]
    | DataFormatJSON
    | DataInfoJSON
    | DTypeName
    | DataAccess
)
"""Available argument types as python types as seen in the JSON."""
ArgumentRetType = (
    str
    | int
    | float
    | list[int]
    | list[int | None]
    | DataFormatJSON
    | DataInfo
    | QualifiedGraphName
    | DTypeName
    | DataAccess
)
"""Available argument types as python types as returned from the argument
parser. This differs from the JSON types in that the types are parsed to the
internal types. Example: A graph name is `str` in the JSON types but
`QualifiedGraphName` returned by the argument parser."""
NodeArguments = dict[str, ArgumentType]
"""A dictionary specifying all JSON node argument values."""


class NodeArg:
    """
    A node arguments parser.
    """
    def __init__(self, ns: GNamespace, name: str, arg: ArgumentType) -> None:
        """
        Creates a node argument.

        Args:
            ns (GNamespace): The namespace.
            name (str): The name of the argument.
            arg (ArgumentType): The argument as it appears in the JSON.
        """
        self._ns = ns
        self._name = name
        self._arg = arg

    def name(self) -> str:
        """
        The argument name.

        Returns:
            str: The name.
        """
        return self._name

    def raw(self) -> ArgumentType:
        """
        The argument as it appears in the JSON.

        Returns:
            ArgumentType: The raw external argument.
        """
        return self._arg

    @overload
    def get(self, type_name: Literal["str"]) -> str:
        ...

    @overload
    def get(self, type_name: Literal["int"]) -> int:
        ...

    @overload
    def get(self, type_name: Literal["float"]) -> float:
        ...

    @overload
    def get(self, type_name: Literal["list_int"]) -> list[int]:
        ...

    @overload
    def get(self, type_name: Literal["shape"]) -> list[int | None]:
        ...

    @overload
    def get(self, type_name: Literal["ushape"]) -> list[int]:
        ...

    @overload
    def get(self, type_name: Literal["format"]) -> DataFormatJSON:
        ...

    @overload
    def get(self, type_name: Literal["info"]) -> DataInfo:
        ...

    @overload
    def get(self, type_name: Literal["graph"]) -> QualifiedGraphName:
        ...

    @overload
    def get(self, type_name: Literal["dtype"]) -> DTypeName:
        ...

    @overload
    def get(self, type_name: Literal["data"]) -> DataAccess:
        ...

    def get(self, type_name: ArgType) -> ArgumentRetType:
        """
        Parses the argument and converts it into the internal type.

        Args:
            type_name (ArgType): The name of the argument type. This should be
                a literal string to make type checking infer the return
                type correctly.

        Raises:
            ValueError: If the argument could not be parsed.

        Returns:
            ArgumentRetType: The argument converted to the internal type.
        """
        val = self._arg
        if type_name == "str":
            return f"{val}"
        if type_name == "graph":
            return QualifiedGraphName(self._ns, GName(f"{val}"))
        if type_name == "dtype":
            return get_dtype_name(f"{val}")
        if type_name == "int":
            return int(cast(int, val))
        if type_name == "float":
            return float(cast(float, val))
        if type_name in ("list_int", "ushape"):
            res_list_int: list[int] = as_int_list(cast(list, val))
            return res_list_int
        if type_name == "shape":
            res_shape: list[int | None] = as_shape(cast(list, val))
            return res_shape
        if type_name == "format":
            res_format: DataFormatJSON = {
                key: (get_dtype_name(dtype), as_shape(dims))
                for key, (dtype, dims) in cast(dict, val).items()
            }
            return res_format
        if type_name == "info":
            dtype, dims = cast(tuple, val)
            res_info: DataInfo = DataInfo(
                get_dtype_name(f"{dtype}"),
                as_shape(cast(list, dims)))
            return res_info
        if type_name == "data":
            offset, length, path = f"{val}".split(NAME_SEP, 2)
            return (path, int(offset), int(length))
        raise ValueError(f"invalid type name {type_name} for arg {self._name}")

    @staticmethod
    def from_node_arguments(ns: GNamespace, args: NodeArguments) -> 'NodeArgs':
        """
        Converts a raw JSON argument dictionary into a dictionary with
        parseable arguments.

        Args:
            ns (GNamespace): The namespace.
            args (NodeArguments): The raw JSON dictionary.

        Returns:
            NodeArgs: Dictionary allowing to parse the arguments.
        """
        return {
            arg_name: NodeArg(ns, arg_name, arg_val)
            for arg_name, arg_val in args.items()
        }

    @staticmethod
    def to_node_arguments(args: 'NodeArgs') -> NodeArguments:
        """
        Convert parseable arguments back to a JSON representation.

        Args:
            args (NodeArgs): The argument dictionary.

        Returns:
            NodeArguments: A JSON argument dictionary.
        """
        return {
            arg_name: arg_val.raw()
            for arg_name, arg_val in args.items()
        }


NodeArgs = dict[str, NodeArg]
"""A dictionary mapping argument names to parseable node arguments."""
