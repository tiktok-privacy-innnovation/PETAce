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

import typing as t
import numpy as np

from petace.backend import PETAceBackendType

from .math import max, min, sum
from .core import SecureArray, support_backends
from .exceptions import AxisError


@support_backends([PETAceBackendType.Duet])
def ptp(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Range of values (maximum - minimum) along an axis.
    The name of the function comes from the acronym for ‘peak to peak’.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    ptp_value : SecureArray
        Range of values along the given axis.
    """
    return max(arr, axis) - min(arr, axis)


@support_backends([PETAceBackendType.Duet])
def average(arr: SecureArray, axis: int = None, weights: t.Union[SecureArray, np.ndarray] = None) -> SecureArray:
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    arr : SecureArray
        Array containing data to be averaged. If a is not an array, a conversion is attempted.

    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    weights : t.Union[SecureArray, np.ndarray], optional
        If `weights=None`, then all data in `arr` are assumed to have a weight equal to one.
        The 1-D calculation is::
            avg = sum(arr * weights) / sum(weights)

    Returns
    -------
    SecureArray
        Return the average along the specified axis.
    """
    if weights is None:
        return mean(arr, axis)
    if not isinstance(weights, (np.ndarray, SecureArray)):
        raise TypeError(f"weights must be SecureArray or np.ndarray, got {type(weights)}")

    if axis is not None and not 0 <= axis < arr.ndim:
        raise AxisError(axis, arr.ndim)

    if axis != 1:
        if weights.shape != arr.shape:
            raise TypeError("Length of weights not compatible with specified axis.")
    elif weights.ndim == 1:
        if isinstance(weights, SecureArray):
            weights = weights.resize(arr.shape)
        else:
            weights = np.resize(arr, arr.shape)
    return sum(arr * weights, axis=axis) / sum(weights, axis=axis)


@support_backends([PETAceBackendType.Duet])
def mean(arr: SecureArray, axis: int = None) -> SecureArray:
    """
    Compute the arithmetic mean along the specified axis.

    Parameters
    ----------
    arr : SecureArray
        Array containing data to be averaged. If a is not an array, a conversion is attempted.

    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.

    Returns
    -------
    SecureArray
        Returns the average of the array elements.
    """
    if axis is not None and not 0 <= axis < arr.ndim:
        raise AxisError(axis, arr.ndim)

    if axis is None:
        total_count = np.prod(arr.shape)
    elif axis == 0:
        total_count = arr.shape[0]
    else:
        total_count = arr.shape[1]
    return sum(arr, axis=axis) / total_count
