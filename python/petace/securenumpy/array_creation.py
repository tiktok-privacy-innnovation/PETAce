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

from typing import Union
import numbers

import numpy as np
from petace.backend import PETAceBackendType
from crypto_data_engine import CSVDataset

from .core import SecureArray, get_engine, support_backends


def array(data: Union[np.ndarray, str, "CSVDataset"], party: int, dtype: np.dtype = np.float64) -> SecureArray:
    """
    Create an SecureArray.

    Parameters
    ----------
    data : np.ndarray
        A numpy array.
    party : int
        Which party provide this data.
    dtype : dtype object, default is np.float64
        Data-type of the array's elements.
        Now only support {np.float64, np.bool_}.

    Returns
    -------
    out : SecureArray
        The SecureArray.
    """
    vm = get_engine()
    if vm.party_id() == party:
        if dtype not in (np.float64, np.bool_):
            raise TypeError(f"Unsupported dtype: {dtype}")
        if isinstance(data, str):
            data = vm.vm.engine_ctx.csv_dataset(file_path=data, partition_num=vm.vm.partition_num)
        if isinstance(data, np.ndarray):
            shape = data.shape
        else:
            shape = data.shape()
        vm.send_shape(shape)
    else:
        shape = vm.recv_shape()

    share_matrix = vm.make_share(data, shape, party, dtype)
    return SecureArray(share_matrix)


@support_backends([PETAceBackendType.Duet])
def fromshare(share: np.ndarray, dtype: np.dtype) -> SecureArray:
    """
    Recover share to SecureArray.

    Parameters
    ----------
    share : np.ndarray
        Share of the new array.
    dtype : np.dtype
        Data type of the new array.

    Returns
    -------
    out : SecureArray
    """
    if not isinstance(share, np.ndarray):
        raise TypeError(f"Only support numpy.ndarray, got {type(share)}")
    if share.dtype != np.int64:
        raise TypeError(f"Only support share with numpy.int64, got {share.dtype}")
    vm = get_engine()
    buffer = vm.new_share(share.shape, dtype, share)
    return SecureArray(buffer)


@support_backends([PETAceBackendType.Duet])
def empty(shape, party: int = 0, dtype: np.dtype = None) -> SecureArray:
    """Return a new array of given shape and vm, without initializing entries.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    party : int
        Which party provide this data.
    dtype
        Data type of the new array. Default is np.float64.

    Returns
    -------
    out : SecureArray
        Array of uninitialized (arbitrary) data of the given shape, dtype.
    """
    data = np.empty(shape, dtype=dtype)
    return array(data, party)


def identity(n: int, party: int = 0, dtype: np.dtype = None) -> SecureArray:
    """
    Return the identity array.
    The identity array is a square array with ones on the main diagonal.

    Parameters
    ----------
    n : int
        The length of the identity array.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array. Default is np.float64.

    Returns
    -------
    out : SecureArray
        n x n array with its main diagonal set to one, and all other elements 0.
    """
    data = np.identity(n, dtype=dtype)
    return array(data, party)


def ones(shape, party: int = 0, dtype: np.dtype = None) -> SecureArray:
    """Return a new array of given shape and vm, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array.

    Returns
    -------
    out : SecureArray
        Array of ones with the given shape.
    """
    data = np.ones(shape, dtype=dtype)
    return array(data, party)


def zeros(shape, party: int = 0, dtype: np.dtype = None) -> SecureArray:
    """
    Return a new array of given shape and vm, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array.

    Returns
    -------
    out : SecureArray
        Array of zeros with the given shape.
    """
    data = np.zeros(shape, dtype)
    return array(data, party)


def full(shape, fill_value: numbers.Number, party: int = 0, dtype: np.dtype = None) -> SecureArray:
    """
    Return a new array of given shape and vm, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    fill_value : number
        Fill value.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array.

    Returns
    -------
    out : SecureArray
        Array of `fill_value` with the given shape.
    """
    data = np.full(shape, fill_value, dtype)
    return array(data, party)


def copy(arr: SecureArray) -> SecureArray:
    """Return a copy of the array.

    Parameters
    ----------
    arr : SecureArray
        The array to be copied.
    Returns
    -------
    out : SecureArray
        A copy of the array.
    """
    return arr.copy()


def arange(start: numbers.Number = None,
           stop: numbers.Number = None,
           step: numbers.Number = None,
           party: int = 0,
           dtype: np.dtype = None) -> SecureArray:
    """Return evenly spaced values within a given interval.

    - arange(stop): Values are generated within the half-open interval [0, stop)
    - arange(start, stop): Values are generated within the half-open interval [start, stop).
    - arange(start, stop, step) Values are generated within the half-open interval [start, stop), with spacing between values given by step.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval. The interval does not include this value, except in case where step is zero.
    step : number
        Spacing between values.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array.
    Returns
    -------
    out : SecureArray
        Array of evenly spaced values.
    """
    if stop is None:
        data = np.arange(start, step=step, dtype=dtype)
    else:
        data = np.arange(start, stop, step=step, dtype=dtype)
    return array(data, party)


def linspace(start, stop, num=50, endpoint=True, party: int = 0, dtype=None, axis=0) -> SecureArray:
    """Returns num evenly spaced samples, calculated over the interval [start, stop].

    Parameters
    ----------
    start : number
        The starting value of the sequence.
    stop : number
        The end value of the sequence.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If `True`, `stop` is the last sample. Otherwise, it is not included.
        Default is `True`.
    party : int
        Which party provide this data.
    dtype : np.dtype
        Data type of the new array.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if `x`
        has more than one dimension. Default is 0.
    Returns
    -------
    samples : ndarray
        There are `num` samples in the dimension specified by `axis`.
    """
    data = np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)
    return array(data, party)


def logspace(start, stop, num=50, endpoint=True, base=10.0, party: int = 0, dtype=None, axis=0) -> SecureArray:
    """Return numbers spaced evenly on a log scale.

    Paramerters
    -----------
    start : number
        The starting value of the sequence.
    stop : number
        The end value of the sequence.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If `True`, `stop` is the last sample. Otherwise, it is not included.
        Default is `True`.
    base : number, optional
        The base of the log space. Default is 10.
    dtype : np.dtype, optional
        The data type of the output array. Default is None.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if `x`
        has more than one dimension. Default is 0.
    Returns
    -------
    out : ndarray
        There are `num` samples in the dimension specified by `axis`.
    """
    data = np.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
    return array(data, party)
