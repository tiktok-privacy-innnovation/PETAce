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

from typing import Tuple

from petace.backend import PETAceBackendType

from .core import SecureArray, get_engine, support_backends
from .exceptions import AxisError


def where(cond: SecureArray, x: SecureArray, y: SecureArray) -> SecureArray:
    """
    Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    cond : SecureArray
        When True, yield x, otherwise yield y.
    x : SecureArray
        Values from which to choose.
    y : SecureArray
        Values from which to choose. x, y and condition need to be broadcastable to some shape.

    Returns
    -------
    out : SecureArray
        An array with elements from x where condition is True, and elements from y elsewhere.
    """
    if cond.shape != x.shape or cond.shape != y.shape:
        raise ValueError("cond, x, y must have the same shape")
    vm = get_engine()
    ret = vm.new_share(x.buffer.shape, x.dtype)
    vm.execute_code("multiplexer", [cond.buffer, y.buffer, x.buffer, ret])
    return SecureArray(ret)


@support_backends([PETAceBackendType.Duet])
def argmax_and_max(arr: SecureArray, axis: int = None) -> Tuple[SecureArray, SecureArray]:
    """
    Returns the indices of the maximum values and maximum values along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    max_index : SecureArray
        Indices of the maximum values along the given axis.
    max_value : SecureArray
        Maximum values along the given axis.

    Notes
    -----
    This function is only supported for axis=0 now.
    """
    if axis is not None and not 0 <= axis < arr.ndim:
        raise AxisError(axis, arr.ndim)
    if axis not in (0, 1, None):
        raise ValueError("argmax only support axis=0 or axis=1")
    if axis is None:
        arr = arr.reshape((-1, 1))
    if axis == 1:
        arr = arr.transpose()
    vm = get_engine()
    if arr.ndim == 2:
        shape = (arr.buffer.shape[1],)
    else:
        shape = ()
    max_index = vm.new_share(shape, arr.dtype)
    max_value = vm.new_share(shape, arr.dtype)
    vm.execute_code("argmax_and_max", [arr.buffer, max_index, max_value])
    return SecureArray(max_index), SecureArray(max_value)


@support_backends([PETAceBackendType.Duet])
def argmin_and_min(arr: SecureArray, axis: int = None) -> Tuple[SecureArray, SecureArray]:
    """
    Returns the indices of the minimum values and minimum values along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    max_index : SecureArray
        Indices of the minimum values along the given axis.
    max_value : SecureArray
        Minimum values along the given axis.

    Notes
    -----
    This function is only supported for axis=1 now.
    """
    min_index, min_value = argmax_and_max(-arr, axis)
    return min_index, -min_value


@support_backends([PETAceBackendType.Duet])
def argmax(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    out : SecureArray
        Indices of the maximum values along the given axis.
    """
    max_index, _ = argmax_and_max(arr, axis)
    return max_index


@support_backends([PETAceBackendType.Duet])
def argmin(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    out : SecureArray
        Indices of the minimum values along the given axis.
    """
    return argmax(-arr, axis)


@support_backends([PETAceBackendType.Duet])
def sort(arr: SecureArray, axis: int = -1) -> SecureArray:
    """
    Returns a sorted copy of an array.

    Parameters
    ----------
    arr : SecureArray
        Input array.

    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    out : SecureArray
        A sorted copy of the array.
    """
    if arr.ndim == 2 and axis is not None:
        raise ValueError("2-d array sort only support axis=None")
    arr_copy = arr.copy()
    arr_copy.quick_sort()
    return arr_copy.flatten()
