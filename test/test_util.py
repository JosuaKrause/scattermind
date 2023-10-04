from typing import Any

import numpy as np
import pytest
import torch

from scattermind.system.info import DataInfo
from scattermind.system.torch_util import (
    create_tensor,
    deserialize_tensor,
    DTypeName,
    extract_shape,
    extract_shapes,
    mask_from_shape,
    mask_from_shapes,
    pad_list,
    pad_tensor,
    same_mask,
    same_shape,
    serialize_tensor,
    set_system_device_cpu,
    str_to_tensor,
    tensor_to_str,
)


NestedList = Any


def test_serialize() -> None:
    set_system_device_cpu()

    def test_ser(mat: np.ndarray, dtype: DTypeName) -> None:
        tensor = create_tensor(mat, dtype)
        other = deserialize_tensor(serialize_tensor(tensor), dtype)
        torch.testing.assert_close(tensor, other)

    test_ser(np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]), "int64")
    test_ser(np.array([
        [[0.1], [0.2]],
        [[0.3], [0.4]],
        [[0.5], [0.6]],
        [[0.7], [0.8]],
    ]), "float32")


def test_pad() -> None:

    def test(
            shape: list[int],
            mat: NestedList,
            padded: NestedList,
            dtype_name: DTypeName) -> None:
        val = create_tensor(np.array(mat), dtype_name)
        out = pad_tensor(val, shape)
        expected = create_tensor(np.array(padded), dtype_name)
        torch.testing.assert_close(out, expected)
        mask = mask_from_shape(list(val.shape), list(out.shape))
        assert list(mask.shape) == list(out.shape)
        shape = extract_shape(mask)
        assert shape == list(val.shape)

    test(
        [2, 3],
        [
            [4, 1],
            [2, 3],
        ],
        [
            [4, 1, 0],
            [2, 3, 0],
        ],
        "int")
    test(
        [3, 2],
        [
            [0.4, 0.1],
            [0.2, 0.3],
        ],
        [
            [0.4, 0.1],
            [0.2, 0.3],
            [0.0, 0.0],
        ],
        "float")
    test(
        [2, 2],
        [
            [0.4, 0.1],
            [0.2, 0.3],
        ],
        [
            [0.4, 0.1],
            [0.2, 0.3],
        ],
        "float64")
    test(
        [3, 2, 2],
        [
            [[4], [1]],
            [[2], [3]],
        ],
        [
            [[4, 0], [1, 0]],
            [[2, 0], [3, 0]],
            [[0, 0], [0, 0]],
        ],
        "int64")
    test(
        [3, 3, 2, 1],
        [
            [[[4]], [[1]]],
            [[[2]], [[3]]],
        ],
        [
            [[[4], [0]], [[1], [0]], [[0], [0]]],
            [[[2], [0]], [[3], [0]], [[0], [0]]],
            [[[0], [0]], [[0], [0]], [[0], [0]]],
        ],
        "int32")


def test_mask() -> None:

    def test(
            own_shape: list[int],
            shape: list[int],
            padded: NestedList) -> None:
        out = mask_from_shape(own_shape, shape)
        expected = create_tensor(np.array(padded), "bool")
        torch.testing.assert_close(out, expected)
        assert extract_shape(out) == own_shape

    test(
        [2, 2],
        [2, 3],
        [
            [True, True, False],
            [True, True, False],
        ])
    test(
        [2, 2],
        [3, 2],
        [
            [True, True],
            [True, True],
            [False, False],
        ])
    test(
        [2, 2],
        [2, 2],
        [
            [True, True],
            [True, True],
        ])
    test(
        [2, 2, 1],
        [3, 2, 2],
        [
            [[True, False], [True, False]],
            [[True, False], [True, False]],
            [[False, False], [False, False]],
        ])
    test(
        [2, 2, 1, 1],
        [3, 3, 2, 1],
        [
            [[[True], [False]], [[True], [False]], [[False], [False]]],
            [[[True], [False]], [[True], [False]], [[False], [False]]],
            [[[False], [False]], [[False], [False]], [[False], [False]]],
        ])


def test_lists() -> None:

    def create(
            arr: NestedList, dtype_name: DTypeName = "float") -> torch.Tensor:
        return create_tensor(np.array(arr), dtype=dtype_name)

    array = [
        create([[[1.0]], [[2.0]]]),
        create([[[3.0, 4.0]], [[5.0, 6.0]]]),
        create([[[7.0, 8.0, 9.0]]]),
    ]
    shapes = [list(val.shape) for val in array]
    info = DataInfo("bool", [None, 1, None])
    max_shape = info.max_shape(shapes)
    assert max_shape == [2, 1, 3]
    padded = pad_list(array, max_shape)
    assert list(padded.shape) == [3, 2, 1, 3]
    torch.testing.assert_close(padded, create([
        [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
        [[[3.0, 4.0, 0.0]], [[5.0, 6.0, 0.0]]],
        [[[7.0, 8.0, 9.0]], [[0.0, 0.0, 0.0]]],
    ]))
    mask = mask_from_shapes(shapes, max_shape)
    assert list(mask.shape) == [3, 2, 1, 3]
    assert same_shape(padded, mask)
    assert not same_shape(padded, create([[1.0, 2.0], [3.0, 4.0]]))
    assert same_mask(mask, create([
        [[[True, False, False]], [[True, False, False]]],
        [[[True, True, False]], [[True, True, False]]],
        [[[True, True, True]], [[False, False, False]]],
    ], "bool"))
    assert extract_shapes(mask) == shapes


def test_str() -> None:

    def rt_str(text: str) -> None:
        assert tensor_to_str(str_to_tensor(text)) == text

    rt_str("test")
    rt_str("this is a long string!")
    rt_str("line\nbreaks")
    rt_str("Hello\u9a6c\u514b😀")


def test_invalid() -> None:
    with pytest.raises(ValueError, match=r"cannot match shapes"):
        mask_from_shape([1, 2, 3], [2, 2])
    with pytest.raises(ValueError, match=r"cannot shrink tensor"):
        mask_from_shape([1, 2, 3], [2, 1, 3])
    with pytest.raises(ValueError, match=r"invalid str from tensor"):
        tensor_to_str(create_tensor(np.array([240, 159, 152, 0]), dtype="int"))
