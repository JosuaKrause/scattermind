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
"""Provides classes that manage information about tensor data."""
from collections.abc import Sequence
from typing import get_args, Literal

import torch

from scattermind.system.base import SessionId
from scattermind.system.helper import DictHelper
from scattermind.system.torch_util import (
    DTypeName,
    get_dtype,
    get_dtype_byte_size,
    get_dtype_name,
)
from scattermind.system.util import as_shape


SpecialDataInfo = Literal["string", "session"]
"""Shortcut data info fields that can be used in the JSON spec."""
UserDataInfoJSON = tuple[DTypeName, Sequence[int | None]] | SpecialDataInfo
"""A user friendly JSON serializable `DataInfo`."""
DataInfoJSON = tuple[DTypeName, Sequence[int | None]]
"""A JSON serializable `DataInfo`."""
UserDataFormatJSON = dict[str, UserDataInfoJSON]
"""A user friendly JSON serializable `DataFormat`."""
DataFormatJSON = dict[str, DataInfoJSON]
"""A JSON serializable `DataFormat`."""

SPECIAL_INFOS: tuple[SpecialDataInfo] = get_args(SpecialDataInfo)
"""All special values for data info."""
STRING_INFO: DataInfoJSON = ("uint8", [None])
"""The info for a string field."""
SESSION_INFO: DataInfoJSON = ("uint8", SessionId.tensor_shape())
"""The info for a session id field."""


def normalize_data_info(info: UserDataInfoJSON) -> DataInfoJSON:
    """
    Convert a user friendly JSON data info into a normalized JSON data info.

    Args:
        info (UserDataInfoJSON): The data info.

    Raises:
        ValueError: If a special value cannot be recognized.

    Returns:
        DataInfoJSON: The normalized JSON data info.
    """
    if not isinstance(info, str):
        return info
    if info == "string":
        return STRING_INFO
    if info == "session":
        return SESSION_INFO
    raise ValueError(
        f"special value of {info=} must be in {SPECIAL_INFOS}")


def denormalize_data_info(info: DataInfoJSON) -> UserDataInfoJSON:
    """
    Replace special data infos with their shortcut representations.

    Args:
        info (DataInfoJSON): The data info.

    Returns:
        UserDataInfoJSON: The user friendly data info.
    """
    if info == STRING_INFO:
        return "string"
    if info == SESSION_INFO:
        return "session"
    return info


def normalize_data_format(data_format: UserDataFormatJSON) -> DataFormatJSON:
    """
    Normalize a user friendly JSON data format.

    Args:
        data_format (UserDataFormatJSON): The data format.

    Returns:
        DataFormatJSON: The normalized data format.
    """
    return {
        key: normalize_data_info(info)
        for key, info in data_format.items()
    }


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
        """
        Whether the given shape are valid for this data info.

        Args:
            shape (list[int]): The shape.

        Returns:
            bool: True if all fixed dimensions match the shape.
        """
        if len(shape) != len(self._dims):
            return False
        for six in self._sdims:
            if shape[six] != self._dims[six]:
                return False
        return True

    def max_shape(self, shapes: list[list[int]]) -> list[int]:
        """
        Computes the shape that can fit all shapes given by the list.
        Fixed dimensions are assumed to be the same (performing this check is
        left to the caller)

        Args:
            shapes (list[list[int]]): The list of shapes.

        Returns:
            list[int]: The shape that can fit all given shapes.
        """
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
                f"expected: {self._dtype} ({get_dtype(self._dtype)}) "
                f"actual: {value.dtype}")
        if not self.valid_shape(list(value.shape)):
            raise ValueError(
                "mismatching shape. "
                f"expected: {self._dims} actual: {value.shape}")
        return value

    def is_uniform(self) -> bool:
        """
        Whether the specified shape is uniform.

        Returns:
            bool: True if there are no variable dimensions.
        """
        return not self._vdims

    @staticmethod
    def from_json(obj: DataInfoJSON) -> 'DataInfo':
        """
        Reads data info from a JSON serializable representation.

        Args:
            obj (DataInfoJSON): The representation to read.

        Returns:
            DataInfo: The data info object.
        """
        dtype, shape = obj
        return DataInfo(get_dtype_name(dtype), as_shape(shape))

    def to_json(self) -> DataInfoJSON:
        """
        Converts the data info into a JSON serializable representation.

        Returns:
            DataInfoJSON: The representation.
        """
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
    def data_format_from_json(obj: UserDataFormatJSON) -> 'DataFormat':
        """
        Reads a data format from a JSON serializable object.

        Args:
            obj (UserDataFormatJSON): The object.

        Returns:
            DataFormat: The data format.
        """
        return DataFormat({
            name: DataInfo.from_json(normalize_data_info(info_obj))
            for name, info_obj in obj.items()
        })

    def data_format_to_json(self) -> UserDataFormatJSON:
        """
        Converts the data info into a JSON serializable representation.

        Returns:
            DataFormatJSON: The JSON serializable representation.
        """
        return {
            name: denormalize_data_info(data_info.to_json())
            for name, data_info in self.items()
        }
