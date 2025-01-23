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
from .math import sum


@support_backends([PETAceBackendType.Duet])
def inner(a: SecureArray, b: SecureArray) -> SecureArray:
    """
    Inner product of two arrays.
    Ordinary inner product of vectors for 1-D arrays (without complex conjugation),
    in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a,b : SecureArray
        If a and b are nonscalar, their last dimensions must match.

    Returns
    -------
    out : SecureArray
        If a and b are both scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned. out.shape = (*a.shape[:-1], *b.shape[:-1])
    """
    if a.ndim == 1 and b.ndim == 1:
        return sum(a * b)
    raise NotImplementedError


@support_backends([PETAceBackendType.Duet])
def dot(a: SecureArray, b: SecureArray) -> SecureArray:
    """
    Dot product of two arrays.

    Parameters
    ----------
    a,b : SecureArray
        If a and b are 1-D arrays, the dot product of the two arrays is returned.
        If a and b are 2-D arrays, the matrix product is returned.
        If a and b are n-D arrays, then the sum product over the last two axes
        is returned.

    Returns
    -------
    out : SecureArray
        The dot product of a and b.
    """
    return a @ b
