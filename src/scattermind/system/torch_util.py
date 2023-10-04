import gzip
import io
from typing import cast, Literal

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


SYS_DEVICE: torch.device | None = None
"""The system device for torch objects."""


def set_system_device(device: torch.device) -> None:
    """
    Sets the system device for torch objects.
    This overwrites automatic detection.

    Args:
        device (torch.device): The desired torch device.
    """
    global SYS_DEVICE

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
    global SYS_DEVICE

    if SYS_DEVICE is None:  # pragma: no cover
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        SYS_DEVICE = device
    return SYS_DEVICE


def create_tensor(mat: np.ndarray, dtype: DTypeName) -> torch.Tensor:
    return torch.tensor(
        mat,
        dtype=get_dtype(dtype),
        device=get_system_device())


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def serialize_tensor(value: torch.Tensor) -> bytes:
    bout = io.BytesIO()
    numpy_type = to_numpy_type(value.dtype)
    with gzip.GzipFile(fileobj=bout, mode="wb") as fout:
        np.save(fout, as_numpy(value).astype(numpy_type))
    return bout.getvalue()


def deserialize_tensor(content: bytes, dtype: DTypeName) -> torch.Tensor:
    binp = io.BytesIO(content)
    with gzip.GzipFile(fileobj=binp, mode="r") as finp:
        return create_tensor(np.load(finp), dtype)


def pad_tensor(value: torch.Tensor, shape: list[int]) -> torch.Tensor:
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
    return pad(value, tuple((
        pad_val
        for right in reversed(padding)
        for pad_val in [0, right]
    )))


def pad_list(
        values: list[torch.Tensor], max_row_shape: list[int]) -> torch.Tensor:
    max_row = [1] + max_row_shape
    return torch.vstack([
        pad_tensor(torch.unsqueeze(value, 0), max_row)
        for value in values
    ])


def mask_from_shape(own_shape: list[int], shape: list[int]) -> torch.Tensor:
    dtype_name: DTypeName = "bool"
    dtype = get_dtype(dtype_name)
    value = create_tensor(
        np.ones(tuple(own_shape), dtype=to_numpy_type(dtype)), dtype_name)
    return pad_tensor(value, shape)


def mask_from_shapes(
        shapes: list[list[int]], max_row_shape: list[int]) -> torch.Tensor:
    max_row = [1] + max_row_shape
    return torch.vstack([
        mask_from_shape([1] + shape, max_row)
        for shape in shapes
    ])


def same_shape(value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
    return list(value_a.shape) == list(value_b.shape)


def same_mask(mask_a: torch.Tensor, mask_b: torch.Tensor) -> bool:
    return same_shape(mask_a, mask_b) and bool((mask_a == mask_b).all().item())


def extract_shapes(mask: torch.Tensor) -> list[list[int]]:
    return [
        extract_shape(mask[ix, :])
        for ix in range(mask.shape[0])
    ]


def extract_shape(mask: torch.Tensor) -> list[int]:
    return list(mask.nonzero().max(0)[0] + 1)


def str_to_tensor(text: str) -> torch.Tensor:
    return create_tensor(
        np.array(list(text.encode("utf-8")), dtype=np.dtype("uint8")),
        "uint8")


def tensor_to_str(value: torch.Tensor) -> str:
    try:
        return bytes(value.ravel().cpu().tolist()).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"invalid str from tensor {value}") from e
