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

from petace.securenumpy import SecureArray, get_engine
from petace.securenumpy import where, ones, zeros


def sigmoid(arr: SecureArray, mode: int = 0) -> SecureArray:
    """Return the sigmoid of an array.

    Parameters
    ----------
    arr : SecureArray
        Input array.
    mode : int, optional
        0: the polynomial approximation of sigmoid, which has higher accuracy but lower speed.
        1: the linear approximation of sigmoid, which has higher speed but low accuracy.
    Returns
    -------
    output : SecureArray
        the sigmoid of an array.
    """
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1.")
    if mode == 0:
        vm = get_engine()
        output = vm.new_share(arr.buffer.shape, arr.dtype)
        vm.execute_code("sigmoid", [arr.buffer, output])
        return SecureArray(output)
    output = arr + 1 / 2
    output = where(output <= 1, output, ones(arr.shape))
    output = where(output >= 0, output, zeros(arr.shape))
    return output
