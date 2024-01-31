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
