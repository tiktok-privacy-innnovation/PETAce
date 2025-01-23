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

import numpy as np

from ._type import PETAceType


class PETAceBuffer:
    """Array objects stored in C ++.

    Attributes
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : np.dtype
        Data type of created array.
    data_type : PETAceType
        Data type of created array.
    reg_addr : int
        Register address of created array.
    """

    def __init__(self, shape: Tuple[int], dtype: np.dtype, data_type: PETAceType, reg_addr: int):
        self.shape = shape
        self.dtype = dtype
        self.data_type = data_type
        self.reg_addr = reg_addr
