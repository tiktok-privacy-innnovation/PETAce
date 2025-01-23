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

import collections
from typing import List, Union, Tuple

from petace.backend import PETAceBackendType

from .core import SecureArray, get_engine, support_backends
from .exceptions import AxisError


@support_backends([PETAceBackendType.Duet])
def vstack(arrays: Union[List[SecureArray], Tuple[SecureArray]]) -> SecureArray:
    """
    Stack arrays in sequence vertically (row wise).

    Parameters
    ----------
    arrays : list of SecureArray
        The arrays must have the same shape along all but the first axis.

    Returns
    -------
    out : SecureArray
        The stacked array.
    """
    if len(arrays) == 0:
        raise ValueError("need at least one array to concatenate")
    if not isinstance(arrays, collections.Iterable):
        raise TypeError("Input must be an iterable")
    vm = get_engine()
    ret = vm.vstack([i.buffer for i in arrays])
    return SecureArray(ret)


@support_backends([PETAceBackendType.Duet])
def hstack(arrays: Union[List[SecureArray], Tuple[SecureArray]]) -> SecureArray:
    """
    Stack arrays in sequence horizontally (column wise).

    Pameters
    ----------
    arrays : list of SecureArray
        The arrays must have the same shape along all but the second axis.

    Returns
    -------
    out : SecureArray
        The stacked array.
    """
    if len(arrays) == 0:
        raise ValueError("need at least one array to concatenate")
    if not isinstance(arrays, collections.Iterable):
        raise TypeError("Input must be an iterable")
    vm = get_engine()
    ret = vm.hstack([i.buffer for i in arrays])
    return SecureArray(ret)


@support_backends([PETAceBackendType.Duet])
def column_stack(tup: Union[List[SecureArray], Tuple[SecureArray]]) -> SecureArray:
    """
    Stack 1-D arrays as columns into a 2-D array.


    Parameters
    ----------
    tup : list of SecureArray
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    out : SecureArray
        The stacked array.
    """
    if not isinstance(tup, collections.Iterable):
        raise TypeError("Input must be an iterable")
    if tup[0].ndim == 1:
        tup = [i.reshape((-1, 1)) for i in tup]
    return hstack(tup)


@support_backends([PETAceBackendType.Duet])
def row_stack(tup: Union[List[SecureArray], Tuple[SecureArray]]) -> SecureArray:
    """
    Stack arrays in sequence vertically (row wise).

    `row_stack` is an alias for vstack. They are the same function.

    Parameters
    ----------
    tup : list of SecureArray
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    out : SecureArray
        The stacked array.
    """
    return vstack(tup)


@support_backends([PETAceBackendType.Duet])
def concatenate(arrays: Union[List[SecureArray], Tuple[SecureArray]], axis: int = 0) -> SecureArray:
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : list of SecureArray
        The arrays must have the same shape along all but the axis.
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    out : SecureArray
        The concatenated array.
    """
    if axis is not None and not 0 <= axis < arrays[0].ndim:
        raise AxisError(axis, arrays[0].ndim)

    if axis is None:
        return row_stack(arrays).flatten()

    if axis == 0:
        if arrays[0].ndim == 0:
            raise ValueError("zero-dimensional arrays cannot be concatenated")
        if arrays[0].ndim == 1:
            return hstack(arrays)
        if arrays[0].ndim == 2:
            return vstack(arrays)
    if axis == 1:
        return hstack(arrays)
    raise ValueError("concatenate only support 0d, 1d, 2d arrays")


@support_backends([PETAceBackendType.Duet])
def reshape(arr: SecureArray, new_shape) -> SecureArray:
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    arr : SecureArray
        Array to be reshaped.
    new_shape : int or tuple of ints
        The new shape should be compatible with the original shape.

    Returns
    -------
    out : SecureArray
        The reshaped array.
    """
    return arr.reshape(new_shape)


@support_backends([PETAceBackendType.Duet])
def resize(arr: SecureArray, new_shape: Union[int, Tuple[int]]) -> SecureArray:
    """
    Return a new array with the specified shape.

    Parameters
    ----------
    arr : SecureArray
        The array to be resized.
    new_shape : int or tuple of ints
        The new shape of the array.

    Returns
    -------
    out : SecureArray
        The resized array.
    """
    return arr.resize(new_shape)
