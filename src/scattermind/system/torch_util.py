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
"""Utility functions for pytorch."""
import gzip
import io
from typing import Any, cast, Literal

import numpy as np
import torch
from torch.nn.functional import pad


DTypeName = Literal[
    "bfloat16",
    "bool",
    "cdouble",
    "cfloat",
    "complex128",
    "complex64",
    "double",
    "float",
    "float16",
    "float32",
    "float64",
    "half",
    "int",
    "int16",
    "int32",
    "int64",
    "int8",
    "long",
    "short",
    "uint8",
]
"""
A supported data type. Names might map to the same internal type
(e.g., `double` vs. `float64`).
"""


DTYPE_MAP: dict[DTypeName, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "cdouble": torch.cdouble,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "complex64": torch.complex64,
    "double": torch.double,
    "float": torch.float,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.half,
    "int": torch.int,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "long": torch.long,
    "short": torch.short,
    "uint8": torch.uint8,
}
"""Mapping from data types to torch dtypes."""


# NOTE: a lot of dtypes will end up mapping to normalized strings
# for example, torch.int will map to "int32" because it is equal to torch.int32
DTYPE_REV_MAP: dict[torch.dtype, DTypeName] = {
    torch.bfloat16: "bfloat16",
    torch.bool: "bool",
    torch.cdouble: "cdouble",
    torch.cfloat: "cfloat",
    torch.complex128: "complex128",
    torch.complex64: "complex64",
    torch.double: "double",
    torch.float: "float",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.half: "half",
    torch.int: "int",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
    torch.long: "long",
    torch.short: "short",
    torch.uint8: "uint8",
}
"""Mapping from torch dtypes to data types."""


TORCH_NUMPY_MAP: dict[torch.dtype, np.dtype] = {
    torch.bfloat16: np.dtype("float16"),
    torch.bool: np.dtype("bool"),
    torch.cdouble: np.dtype("complex128"),
    torch.cfloat: np.dtype("complex64"),
    torch.complex128: np.dtype("complex128"),
    torch.complex64: np.dtype("complex64"),
    torch.double: np.dtype("float64"),
    torch.float: np.dtype("float32"),
    torch.float16: np.dtype("float16"),
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
    torch.half: np.dtype("half"),
    torch.int: np.dtype("int64"),
    torch.int16: np.dtype("int16"),
    torch.int32: np.dtype("int32"),
    torch.int64: np.dtype("int64"),
    torch.int8: np.dtype("int8"),
    torch.long: np.dtype("int64"),
    torch.short: np.dtype("short"),
    torch.uint8: np.dtype("uint8"),
}
"""Mapping from torch dtypes to numpy dtypes."""


TORCH_NUMPY_REV_MAP: dict[np.dtype, torch.dtype] = {
    np.dtype("bool"): torch.bool,
    np.dtype("complex128"): torch.complex128,
    np.dtype("complex64"): torch.complex64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("half"): torch.half,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("int8"): torch.int8,
    np.dtype("short"): torch.short,
    np.dtype("uint8"): torch.uint8,
}
"""Mapping from numpy dtypes to torch dtypes."""


def get_dtype_name(dtype: str) -> DTypeName:
    """
    Converts a string into a `DTypeName`.

    Args:
        dtype (str): The string input.

    Raises:
        ValueError: If the string is not a valid `DTypeName`.

    Returns:
        DTypeName: The dtype name.
    """
    if dtype not in DTYPE_MAP:
        raise ValueError(f"invalid dtype {dtype}")
    return cast(DTypeName, dtype)


def get_dtype(dtype: str) -> torch.dtype:
    """
    Returns the torch dtype for a given data type.

    Args:
        dtype (str): The data type.

    Raises:
        ValueError: If the data type doesn't exist.

    Returns:
        torch.dtype: The torch dtype.
    """
    res = DTYPE_MAP.get(cast(DTypeName, dtype))
    if res is None:
        raise ValueError(f"invalid dtype {dtype}")
    return res


def dtype_to_str(dtype: torch.dtype) -> DTypeName:
    """
    Get the data type for a given torch dtype. Note, that the following
    statement is *not* guaranteed:

    `dtype_to_str(get_dtype(dtype)) == dtype`

    Args:
        dtype (torch.dtype): The torch dtype.

    Raises:
        ValueError: If the torch dtype has no equivalent data type.

    Returns:
        DTypeName: The data type.
    """
    res = DTYPE_REV_MAP.get(dtype)
    if res is None:
        raise ValueError(f"unknown dtype {dtype}")
    return res


SIZE_MAP: dict[DTypeName, int] = {}
"""Mapping of data types to their byte size of the internal representation."""


def get_dtype_byte_size(dtype: DTypeName) -> int:
    """
    Get the byte size of the internal representation of a given data type.

    Args:
        dtype (DTypeName): The data type.

    Returns:
        int: The size of the data type in bytes.
    """
    res = SIZE_MAP.get(dtype)
    if res is None:
        res = get_dtype(dtype).itemsize
        SIZE_MAP[dtype] = res
    return res


def to_numpy_type(dtype: torch.dtype) -> np.dtype:
    """
    Convert a torch dtype to a numpy dtype.

    Args:
        dtype (torch.dtype): The torch dtype.

    Raises:
        ValueError: If there is no corresponding numpy dtype.

    Returns:
        type[np.dtype]: The numpy dtype.
    """
    res = TORCH_NUMPY_MAP.get(dtype)
    if res is None:
        raise ValueError(f"unknown dtype: {dtype}")
    return res


def from_numpy_type(dtype: np.dtype) -> torch.dtype:
    """
    Convert a numpy dtype to a torch dtype.

    Args:
        dtype (np.dtype): The numpy dtype.

    Raises:
        ValueError: If there is no corresponding torch dtype.

    Returns:
        type[torch.dtype]: The torch dtype.
    """
    res = TORCH_NUMPY_REV_MAP.get(dtype)
    if res is None:
        raise ValueError(f"unknown dtype: {dtype}")
    return res


SYS_DEVICE: torch.device | None = None
"""The system device for torch objects."""


def set_system_device(device: torch.device) -> None:
    """
    Sets the system device for torch objects.
    This overwrites automatic detection.

    Args:
        device (torch.device): The desired torch device.
    """
    global SYS_DEVICE  # pylint: disable=global-statement

    SYS_DEVICE = device


def set_system_device_cpu() -> None:
    """
    Sets the system device for torch objects to "cpu".
    """
    set_system_device(torch.device("cpu"))


def get_system_device() -> torch.device:
    """
    Gets the system device for torch objects.
    If no device is set the device is automatically detected.

    Returns:
        torch.device: The system torch device.
    """
    global SYS_DEVICE  # pylint: disable=global-statement

    if SYS_DEVICE is None:  # pragma: no cover
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        SYS_DEVICE = device
    return SYS_DEVICE


def create_tensor(
        mat: np.ndarray | list[Any],
        *,
        dtype: DTypeName | None) -> torch.Tensor:
    """
    Creates a tensor from a numpy array with a given dtype. The tensor will be
    placed in the system device (:py:function::`get_system_device`).

    Args:
        mat (np.ndarray | list[Any]): The tensor content.
        dtype (DTypeName | None): The tensor dtype. If None,
            the type gets inferred from the content. This might not always
            work correctly.

    Returns:
        torch.Tensor: The tensor.
    """
    if dtype is None:
        if isinstance(mat, np.ndarray):
            dtype_obj = from_numpy_type(mat.dtype)
        else:
            dtype = "float"
            arr = mat
            while isinstance(arr, list):
                if len(arr) > 0:
                    arr = arr[0]
                else:
                    break
            if isinstance(arr, int):
                dtype = "int"
            elif not isinstance(arr, (float, list)):
                raise ValueError(f"could not determine dtype for {mat}")
            dtype_obj = get_dtype(dtype)
    else:
        dtype_obj = get_dtype(dtype)
    return torch.tensor(
        mat,
        dtype=dtype_obj,
        device=get_system_device())


def as_numpy(value: torch.Tensor) -> np.ndarray:
    """
    Converts a tensor into a numpy array.

    Args:
        value (torch.Tensor): The tensor.

    Returns:
        np.ndarray: The numpy array.
    """
    return value.detach().cpu().numpy()


def serialize_tensor(value: torch.Tensor) -> bytes:
    """
    Serializes a tensor into a (compressed) byte sequence.

    Args:
        value (torch.Tensor): The tensor to serialize.

    Returns:
        bytes: The serialized byte sequence.
    """
    bout = io.BytesIO()
    numpy_type = to_numpy_type(value.dtype)
    with gzip.GzipFile(fileobj=bout, mode="wb") as fout:
        np.save(fout, as_numpy(value).astype(numpy_type))
    return bout.getvalue()


def deserialize_tensor(content: bytes, dtype: DTypeName) -> torch.Tensor:
    """
    Deserializes a tensor from a byte sequence.

    Args:
        content (bytes): The byte sequence.
        dtype (DTypeName): The expected dtype.

    Returns:
        torch.Tensor: The tensor.
    """
    binp = io.BytesIO(content)
    with gzip.GzipFile(fileobj=binp, mode="r") as finp:
        return create_tensor(np.load(finp), dtype=dtype)


def pad_tensor(value: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """
    Pads a tensor to a given shape. Each dimension where the tensor is smaller
    than the desired shape gets padded with 0s at the end.

    See also :py:function::`pad_list`, :py:function::`mask_from_shape`,
    :py:function::`mask_from_shapes`.

    Args:
        value (torch.Tensor): The tensor to pad.
        shape (list[int]): The desired shape. Each dimension must be either
            equal or larger than the corresponding dimension of the tensor.

    Raises:
        ValueError: If the number of dimensions differs or the desired shape
            has smaller dimensions than the corresponding tensor dimension.

    Returns:
        torch.Tensor: The padded tensor.
    """
    own_shape = list(value.shape)
    if len(own_shape) != len(shape):
        raise ValueError(f"cannot match shapes: {own_shape} {shape}")
    padding: list[int] = []
    for own_dim, dim in zip(own_shape, shape):
        if own_dim > dim:
            raise ValueError(f"cannot shrink tensor: {own_shape} {shape}")
        if own_dim == dim:
            if padding:
                padding.append(0)
            continue
        padding.append(dim - own_dim)
    if not padding:
        return value
    # FIXME check why pylint errors here
    return pad(value, tuple((  # pylint: disable=not-callable
        pad_val
        for right in reversed(padding)
        for pad_val in [0, right]
    )))


def pad_list(
        values: list[torch.Tensor], max_row_shape: list[int]) -> torch.Tensor:
    """
    Combines and pads a list of tensors to the common maximum shape given by
    `max_row_shape`. The resulting tensor has an additional dimension at the
    front where each "row" corresponds to the tensor in the list.

    Example:

    Input:
    values=`[`
    `tensor(shape=[2, 1, 4]),`
    `tensor(shape=[2, 2, 4]),`
    `tensor(shape=[2, 1, 3]),`
    `]`

    max_row_shape=`[2, 2, 4]`

    Output:
    `tensor(shape=[3, 2, 2, 4])`

    See also :py:function::`pad_tensor`, :py:function::`mask_from_shape`,
    :py:function::`mask_from_shapes`.

    Args:
        values (list[torch.Tensor]): The tensors to combine and pad.
        max_row_shape (list[int]): The maximum shape of the tensors.

    Returns:
        torch.Tensor: The combined tensor. Note, that the tensor has an
            additional dimension at the front which corresponds to the length
            of the input list.
    """
    max_row = [1] + max_row_shape
    return torch.vstack([
        pad_tensor(torch.unsqueeze(value, 0), max_row)
        for value in values
    ])


def mask_from_shape(own_shape: list[int], shape: list[int]) -> torch.Tensor:
    """
    Creates a mask for a would be padded tensor.

    See also :py:function::`pad_tensor`, :py:function::`pad_list`,
    :py:function::`mask_from_shapes`.

    Args:
        own_shape (list[int]): The shape of the tensor.
        shape (list[int]): The shape the tensor will be padded to.

    Returns:
        torch.Tensor: The mask which has the padded shape and is True where
            the original tensor has values and False where the original tensor
            would be padded.
    """
    dtype_name: DTypeName = "bool"
    dtype = get_dtype(dtype_name)
    value = create_tensor(
        np.ones(tuple(own_shape), dtype=to_numpy_type(dtype)),
        dtype=dtype_name)
    return pad_tensor(value, shape)


def mask_from_shapes(
        shapes: list[list[int]], max_row_shape: list[int]) -> torch.Tensor:
    """
    Creates a combined mask for a list of would be padded tensors.

    See also :py:function::`pad_tensor`, :py:function::`pad_list`,
    :py:function::`mask_from_shape`.

    Args:
        shapes (list[list[int]]): The shapes of the tensors.
        max_row_shape (list[int]): The shape the tensors will be padded to.

    Returns:
        torch.Tensor: The mask which has the padded shape and is True where
            the original tensors have values and False where the original
            tensors would be padded. Note, that the tensor has an
            additional dimension at the front which corresponds to the length
            of the input list.
    """
    max_row = [1] + max_row_shape
    return torch.vstack([
        mask_from_shape([1] + shape, max_row)
        for shape in shapes
    ])


def same_shape(value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
    """
    Whether two tensors have the same shapes.

    Args:
        value_a (torch.Tensor): The first tensor.
        value_b (torch.Tensor): The second tensor.

    Returns:
        bool: True if both tensors have the same shape.
    """
    return list(value_a.shape) == list(value_b.shape)


def same_mask(mask_a: torch.Tensor, mask_b: torch.Tensor) -> bool:
    """
    Whether two masks are the same.

    See also :py:function::`mask_from_shape`, :py:function::`mask_from_shapes`.

    Args:
        mask_a (torch.Tensor): The first mask.
        mask_b (torch.Tensor): The second mask.

    Returns:
        bool: True if both masks have the same shape and mask the same values.
    """
    return same_shape(mask_a, mask_b) and bool((mask_a == mask_b).all().item())


def extract_shapes(mask: torch.Tensor) -> list[list[int]]:
    """
    Extracts the individual shapes from a combined mask tensor. The first
    dimension represents the rows. Masks can only have False values at the end
    of a dimension.

    See also :py:function::`mask_from_shapes`.

    Args:
        mask (torch.Tensor): The combined mask tensor.

    Returns:
        list[list[int]]: A list of tensor shapes corresponding to the rows.
            The length of the list equals the number of rows (i.e., the size
            of the first dimensions of the mask).
    """
    return [
        extract_shape(mask[ix, :])
        for ix in range(mask.shape[0])
    ]


def extract_shape(mask: torch.Tensor) -> list[int]:
    """
    Extracts the actual shape from the mask. This assumes that False values
    can only be at the end of each dimension.

    Example:

    mask=`[[True, True, False], [True, True, False], [False, False, False]]`
    (shape=[3, 3])

    output=`[2, 2]`

    Args:
        mask (torch.Tensor): The mask.

    Returns:
        list[int]: The shape given by which values are unmasked.
    """
    return list((mask.nonzero().max(0)[0] + 1).cpu().numpy())


def str_to_tensor(text: str) -> torch.Tensor:
    """
    Convert a string into a one dimensional tensor using UTF-8 encoding.

    Args:
        text (str): The text.

    Returns:
        torch.Tensor: A one dimensional uint8 tensor containing the UTF-8
            byte values of the given text.
    """
    return create_tensor(
        np.array(list(text.encode("utf-8")), dtype=np.dtype("uint8")),
        dtype="uint8")


def tensor_to_str(value: torch.Tensor) -> str:
    """
    Convert a tensor into a string. The tensor is flattened and values are
    interpreted as UTF-8 bytes.

    Args:
        value (torch.Tensor): The tensor.

    Raises:
        ValueError: If the tensor could not be converted to a string.

    Returns:
        str: The string.
    """
    try:
        return bytes(value.ravel().cpu().tolist()).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"invalid str from tensor {value}") from e
