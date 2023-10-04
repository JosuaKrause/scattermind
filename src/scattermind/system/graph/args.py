from typing import cast, Literal, overload

from scattermind.system.info import DataFormatJSON, DataInfo, DataInfoJSON
from scattermind.system.names import GName, NAME_SEP
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
ArgumentRetType = (
    str
    | int
    | float
    | list[int]
    | list[int | None]
    | DataFormatJSON
    | DataInfo
    | GName
    | DTypeName
    | DataAccess
)
NodeArguments = dict[str, ArgumentType]


class NodeArg:
    def __init__(self, name: str, arg: ArgumentType) -> None:
        self._name = name
        self._arg = arg

    def name(self) -> str:
        return self._name

    def raw(self) -> ArgumentType:
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
    def get(self, type_name: Literal["graph"]) -> GName:
        ...

    @overload
    def get(self, type_name: Literal["dtype"]) -> DTypeName:
        ...

    @overload
    def get(self, type_name: Literal["data"]) -> DataAccess:
        ...

    def get(self, type_name: ArgType) -> ArgumentRetType:
        val = self._arg
        if type_name == "str":
            return f"{val}"
        if type_name == "graph":
            return GName(f"{val}")
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
    def from_node_arguments(args: NodeArguments) -> 'NodeArgs':
        return {
            arg_name: NodeArg(arg_name, arg_val)
            for arg_name, arg_val in args.items()
        }

    @staticmethod
    def to_node_arguments(args: 'NodeArgs') -> NodeArguments:
        return {
            arg_name: arg_val.raw()
            for arg_name, arg_val in args.items()
        }


NodeArgs = dict[str, NodeArg]
