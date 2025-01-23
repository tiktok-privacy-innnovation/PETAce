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

from petace.backend import PETAceBackendType

from .core import SecureArray, support_backends
from .sort_search import argmax_and_max


@support_backends([PETAceBackendType.Duet])
def sum(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    arr : SecureArray
        Elements to sum.
    axis : None or int {0, 1}, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.

    Returns
    -------
    sum_along_axis : SecureArray
        An array with the same shape as `arr`, with the specified
        axis removed.   If `arr` is a 0-d array, or if `axis` is None, a scalar
        is returned.
    """
    if arr.ndim == 0:
        return arr.copy()
    res = 0
    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            res += arr[i]
    elif axis == 0:
        for i in range(arr.shape[0]):
            res += arr[i]
    elif axis == 1:
        for i in range(arr.shape[1]):
            res += arr[:, i]
    else:
        res = sum(arr, axis=0)
        res = sum(res)
    return res


@support_backends([PETAceBackendType.Duet])
def max(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Return the maximum of an array or the maximum along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    max_value : SecureArray
        Maximum values along the given axis.
    """
    _, max_value = argmax_and_max(arr, axis)
    return max_value


@support_backends([PETAceBackendType.Duet])
def min(arr: SecureArray, axis: int = None) -> SecureArray:
    """Return the minimum of an array or the minimum along an axis.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    min_value : SecureArray
        Minimum values along the given axis.
    """
    _, min_value = argmax_and_max(-arr, axis)
    return -min_value


@support_backends([PETAceBackendType.Duet])
def prod(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    arr : SecureArray
       Input data.
    axis : None or int {0, 1}, optional
        Axis or axes along which a product is performed.

    Returns
    -------
    prod_along_axis : SecureArray
        An array
    """
    if arr.ndim == 0:
        return arr.copy()

    res = 1
    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            res *= arr[i]
    elif axis == 0:
        for i in range(arr.shape[0]):
            res *= arr[i]
    elif axis == 1:
        for i in range(arr.shape[1]):
            res *= arr[:, i]
    else:
        res = prod(arr, axis=0)
        res = prod(res)
    return res
