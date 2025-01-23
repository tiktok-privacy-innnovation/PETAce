# Copyright 2023 TikTok Pte. Ltd.
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

from __future__ import annotations
import numbers
from typing import Tuple, Union

import numpy as np

from petace.engine import PETAceBuffer
from petace.backend.bigdata.vm import BigDataVM
from petace.backend import PETAceBackendType
from .index_utils import index_to_block_index
from .broad_cast import auto_broadcast
from .init import get_engine
from .support_backend import support_backends


class SecureArray:
    """
    An SecureArray is a multidimensional container of items of the same type and size.

    The number of dimensions and items in an array is defined by its shape, which is a tuple of N non-negative
    integers that specify the sizes of each dimension. The type of items in the array is specified by a separate
    data-type object (dtype), one of which is associated with each ndarray.

    Parameters
    ----------
    buffer : PETAceBuffer
        The buffer of the new array.

    Attributes
    ----------
    dtype : dtype object
        Data-type of the array's elements.
    size : int
        Number of elements in the array.
    ndim : int
        Number of array dimensions.
    shape : tuple of ints
        The dimensions of the array.
    """
    # Higher values are used to suppress the numpy overload priority.
    __array_priority__ = 10000

    def __init__(self, buffer: PETAceBuffer) -> None:
        self.buffer = buffer
        self.engine = get_engine()

    @property
    def shape(self) -> Tuple[int]:
        """Tuple of array dimensions."""
        return self.buffer.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the array's elements."""
        return self.buffer.dtype

    @property
    def size(self) -> int:
        """Number of elements in the array.
        Equal to np.prod(a.shape), i.e., the product of the array's dimensions.
        """
        if self.ndim == 0:
            return 1
        return np.prod(self.shape)

    @property
    def T(self) -> SecureArray:
        """Transpose of the array."""
        return self.transpose()

    def __check_type(self, obj, expect_type):
        if not isinstance(obj, expect_type):
            raise TypeError(f"Expect {expect_type}, got {type(obj)}")

    def __len__(self) -> int:
        """Return len(self)."""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __repr__(self):
        """Return repr(self)."""
        return f"SecureArray(shape={self.shape}, dtype={self.dtype}, party={self.engine.party_id()}, addr={self.buffer.reg_addr})"

    def reveal_to(self, party: int, to_numpy: bool = True) -> np.ndarray:
        """
        Reveal a SecureArray to a given party. Another party will get None.

        Parameters
        ----------
        party : int
            The party to reveal to.
        to_numpy: bool
            The default setting for converting the result to a numpy array is True.

        Returns
        -------
        out: np.ndarray
            The revealed array.
        """
        private = self.engine.new_private(self.shape, self.dtype, None, party)
        self.engine.execute_code("reveal", [self.buffer, private])
        if isinstance(self.engine.vm, BigDataVM):
            self.engine.vm.run()
            self.engine.vm.refine_engine_ctx()
            if not to_numpy:
                return private.reg_addr
        res = self.engine.to_numpy(private)
        self.engine.delete_buffer(private)
        if self.engine.party_id() == party:
            res = res.reshape(self.shape)
        return res

    def to_share(self) -> np.ndarray[np.int64]:
        """Share the SecureArray to different party.
        You can use snp.fromshare to recover a SecureArray from a numpy share array.

        Returns
        -------
        out: np.ndarray[np.int64]
            The shared array.
        """
        share = self.engine.to_share(self.buffer)
        return share.reshape(self.shape)

    def __del__(self):
        self.engine.delete_buffer(self.buffer)

    def __getitem__(self, index: Union[slice, int]) -> SecureArray:
        if not isinstance(index, (int, slice, tuple)):
            raise IndexError(f"unsupported index type {type(index)}")

        shape = self.shape
        if self.ndim == 1:
            shape = (shape[0], 1)
        row_start, col_start, row_num, col_num = index_to_block_index(index, shape, self.ndim)
        new_shape = np.empty(self.shape)[index].shape
        ret = self.engine.get_matrix_block(self.buffer, new_shape, row_start, col_start, row_num, col_num)
        # transform shape of 1d matrix to (1, n) in cpp
        if len(ret.shape) == 1 and self._debug_shape()[0] != 1:
            ret = self.engine.inner_flatten(ret)
        return SecureArray(ret)

    def __setitem__(self, key: Union[slice, int, tuple], value: Union[numbers.Number, np.ndarray, SecureArray]) -> None:
        if not isinstance(key, (int, slice, tuple)):
            raise IndexError(f"unsupported index type {type(key)}")
        self.__check_type(value, (bool, numbers.Number, np.ndarray, SecureArray))
        if self.ndim not in (1, 2):
            raise ValueError(f"Unsupported dimension {self.ndim}")

        row_start, col_start, row_number, col_number = index_to_block_index(key, self.shape, self.ndim)
        block_shape = (row_number, col_number)

        if isinstance(value, numbers.Number):
            value = np.resize(value, block_shape).astype(self.dtype)
        if isinstance(value, np.ndarray):
            if value.size != np.prod(block_shape):
                raise ValueError(f"Except {np.prod(block_shape)} but got {value.size} elements")
            value = np.array(value).reshape(block_shape)
            share_matrix = self.engine.make_share(value, block_shape, 0, self.dtype)
            value = SecureArray(share_matrix)

        row_start_public = self.engine.new_public(row_start)
        col_start_public = self.engine.new_public(col_start)
        row_number_public = self.engine.new_public(row_number)
        col_number_public = self.engine.new_public(col_number)

        self.engine.execute_code(
            "set_item",
            [value.buffer, self.buffer, row_start_public, col_start_public, row_number_public, col_number_public])
        self.engine.delete_buffer(row_start_public)
        self.engine.delete_buffer(col_start_public)
        self.engine.delete_buffer(row_number_public)
        self.engine.delete_buffer(col_number_public)

    def __operator(self, arr1: SecureArray, arr2: Union[SecureArray, np.ndarray], operation: str) -> SecureArray:
        if operation in {"lt", "gt", "ge", "eq", "ne", "and", "or", "xor"}:
            res_type = np.bool_
        else:
            res_type = self.dtype
        if arr1.shape != arr2.shape:
            raise ValueError(f"Cannot {operation} two arrays with different shape")
        share_res = self.engine.new_share(arr1.shape, res_type)
        if isinstance(arr2, SecureArray):
            self.engine.execute_code(operation, [arr1.buffer, arr2.buffer, share_res])
        elif isinstance(arr2, np.ndarray):
            arr2 = arr2.astype(self.dtype)
            public = self.engine.new_public(arr2)
            self.engine.execute_code(operation, [arr1.buffer, public, share_res])
            self.engine.delete_buffer(public)
        else:
            raise TypeError(f"Unsupported data type {type(arr2)}")
        return SecureArray(share_res)

    def __add__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self+other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "add")

    def __sub__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self-other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "sub")

    def __mul__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self*other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "mul")

    def __radd__(self, other: Union[numbers.Number, np.ndarray]) -> SecureArray:
        """Return other + self."""
        self.__check_type(other, (numbers.Number, np.ndarray))
        return self + other

    def __rsub__(self, other: Union[numbers.Number, np.ndarray]) -> SecureArray:
        """Return other-self."""
        self.__check_type(other, (numbers.Number, np.ndarray))
        return -self + other

    def __rmul__(self, other: Union[numbers.Number, np.ndarray]) -> SecureArray:
        """Return other*self."""
        self.__check_type(other, (numbers.Number, np.ndarray))
        return self * other

    def __rtruediv__(self, other: Union[numbers.Number, np.ndarray]) -> SecureArray:
        """Return other/self
        """
        self.__check_type(other, (numbers.Number, np.ndarray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return arr2 / arr1

    def __truediv__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self/other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "div")

    def __lt__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self<other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "lt")

    def __gt__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self>other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "gt")

    def __eq__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self==other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        return self.__operator(arr1, arr2, "eq")

    def __ne__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self!=other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        return ~(self == other)

    def __le__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self<=other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return arr2 >= arr1

    def __ge__(self, other: Union[numbers.Number, np.ndarray, SecureArray]) -> SecureArray:
        """Return self>=other."""
        self.__check_type(other, (numbers.Number, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return self.__operator(arr1, arr2, "ge")

    def __neg__(self) -> SecureArray:
        """-self"""
        return self * -1

    def __invert__(self) -> SecureArray:
        """~self"""
        if not self.dtype == np.bool_:
            raise TypeError("~ only support bool type")
        ret = self.engine.new_share(self.shape, np.bool_)
        self.engine.execute_code("not", [self.buffer, ret])
        return SecureArray(ret)

    def quick_sort(self) -> None:
        """Sort all elements inplace and not change shape.
        Equal to `np.sort(data, axis=None).reshape(data.shape)`
        """
        self.engine.execute_code("quick_sort", [self.buffer])

    def quick_sort_by_column(self, column_index: int) -> SecureArray:
        """Sort all elements according to the column_index.
        Equal to `data[np.argsort(data[:, column_index])]`
        """
        self.__check_type(column_index, int)
        ret = self.engine.new_share(self.shape, self.dtype)
        column_index = self.engine.new_public(int(column_index))
        self.engine.execute_code("quick_sort", [self.buffer, column_index, ret])
        self.engine.delete_buffer(column_index)
        return SecureArray(ret)

    def __and__(self, other: Union[bool, np.ndarray, SecureArray]) -> SecureArray:
        """Return self & other."""
        self.__check_type(other, (bool, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return self.__operator(arr1, arr2, "and")

    def __or__(self, other: Union[bool, np.ndarray, SecureArray]) -> SecureArray:
        """Return self | other."""
        self.__check_type(other, (bool, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return self.__operator(arr1, arr2, "or")

    def __xor__(self, other: Union[bool, np.ndarray, SecureArray]) -> SecureArray:
        """Return self ^ other."""
        self.__check_type(other, (bool, np.ndarray, SecureArray))
        arr1, arr2 = auto_broadcast(self, other)
        if isinstance(arr2, np.ndarray):
            arr2 = SecureArray(self.engine.make_share(arr2, arr2.shape, 0, self.dtype))
        return self.__operator(arr1, arr2, "xor")

    def __rand__(self, other: Union[bool, np.ndarray]) -> SecureArray:
        """Return other & self."""
        self.__check_type(other, (bool, np.ndarray))
        return self & other

    def __ror__(self, other: Union[bool, SecureArray]) -> SecureArray:
        """Return self | other."""
        self.__check_type(other, (bool, np.ndarray))
        return self | other

    def __rxor__(self, other: Union[bool, SecureArray]) -> SecureArray:
        """Return self ^ other."""
        self.__check_type(other, (bool, np.ndarray))
        return self ^ other

    def __copy__(self) -> SecureArray:
        """Used if copy.copy is called on an array. Returns a copy of the array.
        """
        return self.copy()

    def copy(self) -> SecureArray:
        """Return copy of array.
        """
        return self + 0

    @support_backends([PETAceBackendType.Duet])
    def __matmul__(self, other: Union[np.ndarray, SecureArray]) -> SecureArray:
        """return self @ other
        """
        self.__check_type(other, (np.ndarray, SecureArray))
        res_shape = (np.ones(self.shape) @ np.ones(other.shape)).shape
        share_res = self.engine.new_share(res_shape, self.dtype)
        if other.ndim == 1:
            other = other.reshape((-1, 1))

        if isinstance(other, SecureArray):
            self.engine.execute_code("mat_mul", [self.buffer, other.buffer, share_res])
        else:
            other = other.astype(self.dtype)
            public = self.engine.new_public(other)
            self.engine.execute_code("mat_mul", [self.buffer, public, share_res])
            self.engine.delete_buffer(public)

        if len(res_shape) == 1 and self._debug_shape()[0] != 1:
            share_res = self.engine.inner_flatten(share_res)
        return SecureArray(share_res)

    @support_backends([PETAceBackendType.Duet])
    def __rmatmul__(self, other: np.ndarray) -> SecureArray:
        """return other @ self
        """
        self.__check_type(other, np.ndarray)
        res_shape = (np.ones(other.shape) @ np.ones(self.shape)).shape
        other = other.astype(self.dtype)
        if other.ndim == 1 and self.ndim == 1:
            return self @ other.reshape((-1, 1))
        share_res = self.engine.new_share(res_shape, self.dtype)
        if self.ndim == 1:
            other = other.transpose()
            public = self.engine.new_public(other)
            self_trans = self.reshape((1, -1))
            self.engine.execute_code("mat_mul", [self_trans.buffer, public, share_res])
        else:
            public = self.engine.new_public(other)
            self.engine.execute_code("mat_mul", [public, self.buffer, share_res])
        self.engine.delete_buffer(public)
        return SecureArray(share_res)

    @support_backends([PETAceBackendType.Duet])
    def reshape(self, new_shape) -> SecureArray:
        """
        Gives a new shape to an array without changing its data.

        Parameters
        ----------
        new_shape : int or tuple of ints
            _description_

        Returns
        -------
        SecureArray
            _description_
        """
        if not isinstance(new_shape, (int, tuple)):
            raise TypeError(f"Unsupported shape type {type(new_shape)}")
        if isinstance(new_shape, int):
            if new_shape == -1:
                new_shape = (self.size,)
            else:
                new_shape = (new_shape,)

        if len(new_shape) > 2:
            raise ValueError("Cannot reshape array exceed 2-d")
        if len(new_shape) == 0:
            col = 1
            row = 1
            new_shape = ()
        elif len(new_shape) == 1:
            row = 1
            col = new_shape[0]
            if col == -1:
                col = self.size
                new_shape = (col,)
        else:
            row = new_shape[0]
            col = new_shape[1]
            if row == -1:
                row = self.size // col
            elif col == -1:
                col = self.size // row
            new_shape = (row, col)

        if self.size != np.prod([row, col]):
            raise ValueError(f"cannot reshape array of size {self.size} into shape {new_shape}")

        ret = self.engine.new_share(new_shape, self.dtype)

        row_public = self.engine.new_public(int(row))
        col_public = self.engine.new_public(int(col))
        self.engine.execute_code("reshape", [self.buffer, row_public, col_public, ret])
        self.engine.delete_buffer(row_public)
        self.engine.delete_buffer(col_public)
        return SecureArray(ret)

    @support_backends([PETAceBackendType.Duet])
    def flatten(self) -> SecureArray:
        """Return a copy of the array collapsed into one dimension.
        """
        return self.reshape(-1)

    @support_backends([PETAceBackendType.Duet])
    def transpose(self) -> SecureArray:
        """Returns a copy of the array with axes transposed.
        """
        if self.ndim < 2:
            return self + 0
        ret = self.engine.new_share((self.shape[1], self.shape[0]), self.dtype)
        self.engine.execute_code("transpose", [self.buffer, ret])
        return SecureArray(ret)

    @support_backends([PETAceBackendType.Duet])
    def resize(self, new_shape) -> SecureArray:
        """Change shape and size of array and return a copy.
        """
        if not isinstance(new_shape, (int, tuple)):
            raise TypeError(f"Unsupported shape type {type(new_shape)}")
        if isinstance(new_shape, int) and new_shape < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")
        if isinstance(new_shape, tuple) and np.any(np.array(new_shape) < 0):
            raise ValueError("all elements of `new_shape` must be non-negative")

        if isinstance(new_shape, int):
            row = 1
            col = new_shape
            new_shape = (new_shape,)
        elif len(new_shape) == 1:
            row = 1
            col = new_shape[0]
        else:
            row = new_shape[0]
            col = new_shape[1]

        ret = self.engine.new_share(new_shape, self.dtype)
        row_public = self.engine.new_public(row)
        col_public = self.engine.new_public(col)
        self.engine.execute_code("resize", [self.buffer, row_public, col_public, ret])
        self.engine.delete_buffer(row_public)
        self.engine.delete_buffer(col_public)
        return SecureArray(ret)

    @support_backends([PETAceBackendType.Duet])
    def ravel(self) -> SecureArray:
        """Return a flattened array.
        """
        return self.flatten()

    def _debug_shape(self):
        """shape of inner matrix.
        Just for debug
        """
        return self.engine.get_debug_shape(self.buffer)
