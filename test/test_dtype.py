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
"""Test dtypes."""
from typing import get_args

import pytest

from scattermind.system.torch_util import (
    dtype_to_str,
    DTypeName,
    get_dtype,
    get_dtype_byte_size,
    get_dtype_name,
    to_numpy_type,
)


def test_dtype() -> None:
    """Test dtypes."""
    for dtype in get_args(DTypeName):
        torch_dtype = get_dtype(dtype)
        other_dtype = get_dtype(dtype_to_str(torch_dtype))
        assert torch_dtype == other_dtype
        assert get_dtype_byte_size(dtype) == torch_dtype.itemsize
        np_dtype = to_numpy_type(torch_dtype)
        assert get_dtype_byte_size(dtype) == np_dtype.itemsize
        kind = np_dtype.kind
        assert torch_dtype.is_complex == ("c" in kind)
        assert torch_dtype.is_floating_point == ("f" in kind)
        assert torch_dtype.is_signed == (
            "i" in kind or "f" in kind or "c" in kind)
        assert not torch_dtype.is_signed == ("u" in kind or "b" in kind)


def test_invalid_dtype() -> None:
    """Test invalid dtypes."""
    with pytest.raises(ValueError, match=r"invalid dtype"):
        get_dtype("foo")
    with pytest.raises(ValueError, match=r"invalid dtype"):
        get_dtype_name("foo")
    with pytest.raises(ValueError, match=r"unknown dtype"):
        dtype_to_str(None)  # type: ignore
    with pytest.raises(ValueError, match=r"unknown dtype"):
        to_numpy_type(None)  # type: ignore
