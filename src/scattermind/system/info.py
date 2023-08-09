from collections.abc import Sequence

import torch

from scattermind.system.helper import DictHelper
from scattermind.system.torch_util import (
    DTypeName,
    get_dtype,
    get_dtype_byte_size,
    get_dtype_name,
)
from scattermind.system.util import as_shape


DataInfoJSON = tuple[DTypeName, Sequence[int | None]]
"""A JSON serializable `DataInfo`."""
DataFormatJSON = dict[str, DataInfoJSON]
"""A JSON serializable `DataFormat`."""


class DataInfo:
    """
    Describes the shape and type of data.
    """
    def __init__(self, dtype: DTypeName, dims: Sequence[int | None]) -> None:
        """
        Create a `DataInfo` object.

        Args:
            dtype (DTypeName): The type of the data.
            dims (Sequence[int | None]): The shape of the data.
        """
        self._dtype = dtype
        self._dims = dims
        self._vdims = {ix for ix in range(len(dims)) if dims[ix] is None}
        self._sdims = {ix for ix in range(len(dims)) if dims[ix] is not None}

    def shape(self) -> list[int | None]:
        """
        Returns the shape of the data.

        Returns:
            list[int | None]: The shape of the data.
        """
        return list(self._dims)

    def valid_shape(self, shape: list[int]) -> bool:
        if len(shape) != len(self._dims):
            return False
        for six in self._sdims:
            if shape[six] != self._dims[six]:
                return False
        return True

    def max_shape(self, shapes: list[list[int]]) -> list[int]:
        max_shape = [0 if dim is None else dim for dim in self._dims]
        for vdim in self._vdims:
            for shape in shapes:
                max_shape[vdim] = max(max_shape[vdim], shape[vdim])
        return max_shape

    def dtype(self) -> DTypeName:
        """
        Returns the type of the data.

        Returns:
            DTypeName: The data type.
        """
        return self._dtype

    @staticmethod
    def item_count(max_shape: list[int]) -> int:
        """
        How many elements the data contains.

        Args:
            max_shape (list[int]):
                The maximum shape a collection of this type has.
                This should be computed for each group of this data type anew.

        Returns:
            int: The product of the size of all dimensions.
        """
        count = 1
        for dim in max_shape:
            count *= dim
        return count

    def byte_size(self, max_shape: list[int]) -> int:
        """
        The number of bytes required to store the raw data.

        Args:
            max_shape (list[int]):
                The maximum shape a collection of this type has.
                This should be computed for each group of this data type anew.

        Returns:
            int:
                The product of the size of all dimensions times the
                byte size of the data type.
        """
        return self.item_count(max_shape) * get_dtype_byte_size(self._dtype)

    def check_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """
        Ensures that the given tensor matches the data info. The data type
        and the shape must be the same.

        Args:
            value (torch.Tensor): The tensor to inspect.

        Raises:
            ValueError: If the data type of the shape does not match.

        Returns:
            torch.Tensor: The verified tensor.
        """
        if value.dtype != get_dtype(self._dtype):
            raise ValueError(
                "mismatching dtype. "
                f"expected: {self._dtype} actual: {value.dtype}")
        if not self.valid_shape(list(value.shape)):
            raise ValueError(
                "mismatching shape. "
                f"expected: {self._dims} actual: {value.shape}")
        return value

    def is_uniform(self) -> bool:
        return not self._vdims

    @staticmethod
    def from_json(obj: DataInfoJSON) -> 'DataInfo':
        dtype, shape = obj
        return DataInfo(get_dtype_name(dtype), as_shape(shape))

    def to_json(self) -> DataInfoJSON:
        return (self.dtype(), self.shape())

    def __eq__(self, other: object) -> bool:
        """
        Tests whether two data infos are the same.

        Args:
            other (object): The other data info.

        Returns:
            bool: Data infos are equal if the data type and shape matches.
        """
        if self is other:
            return True
        if not isinstance(other, DataInfo):
            return False
        if get_dtype(self.dtype()) != get_dtype(other.dtype()):
            return False
        return self.shape() != other.shape()

    def __ne__(self, other: object) -> bool:
        """
        Tests whether two data infos are different.

        Args:
            other (object): The other data info.

        Returns:
            bool: Data infos are different if the data type or shape differ.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        The hash of a data info object.

        Returns:
            int: The hash depends on the data type and shape.
        """
        return hash(get_dtype(self.dtype())) + 31 * hash(self.shape())

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.dtype()},{self.shape()}]"

    def __repr__(self) -> str:
        return self.__str__()


class DataFormat(DictHelper[DataInfo]):
    """
    A `DataFormat` is a collection of named `DataInfo` formats.
    Methods to work with `DataFormat`s can be found as static methods of
    `DataInfo`.
    """
    @staticmethod
    def data_format_from_json(obj: DataFormatJSON) -> 'DataFormat':
        return DataFormat({
            name: DataInfo.from_json(info_obj)
            for name, info_obj in obj.items()
        })

    def data_format_to_json(self) -> DataFormatJSON:
        return {
            name: data_info.to_json()
            for name, data_info in self.items()
        }
