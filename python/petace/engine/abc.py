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

import numbers
from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import numpy as np

from .buffer import PETAceBuffer


class PETAceEngineABC(ABC):

    @abstractmethod
    def party_id(self) -> int:
        pass

    @abstractmethod
    def new_private(self, shape: Tuple[int], dtype: np.dtype, data: np.ndarray, party: int) -> PETAceBuffer:
        pass

    @abstractmethod
    def new_share(self, shape: Tuple[int], dtype: np.dtype, share: np.ndarray = None) -> PETAceBuffer:
        pass

    @abstractmethod
    def new_public(self, data: Union[numbers.Number, np.ndarray]) -> PETAceBuffer:
        pass

    @abstractmethod
    def delete_buffer(self, obj: PETAceBuffer):
        pass

    @abstractmethod
    def to_numpy(self, obj: PETAceBuffer) -> np.ndarray:
        pass

    @abstractmethod
    def to_share(self, obj: PETAceBuffer) -> np.ndarray:
        pass

    @abstractmethod
    def execute_code(self, operation: str, objs: List[PETAceBuffer]) -> None:
        pass

    @abstractmethod
    def vstack(self, buffers: Union[List[PETAceBuffer], Tuple[PETAceBuffer]]) -> PETAceBuffer:
        pass

    @abstractmethod
    def hstack(self, buffers: Union[List[PETAceBuffer], Tuple[PETAceBuffer]]) -> PETAceBuffer:
        pass

    @abstractmethod
    def make_share(self,
                   data: Union[np.ndarray, None],
                   shape,
                   party: int = 0,
                   dtype: np.dtype = np.float64) -> PETAceBuffer:
        pass

    @abstractmethod
    def send_shape(self, shape: tuple):
        pass

    @abstractmethod
    def recv_shape(self) -> tuple:
        pass

    @abstractmethod
    def inner_flatten(self, buffer: PETAceBuffer) -> PETAceBuffer:
        pass

    @abstractmethod
    def get_debug_shape(self, buffer: PETAceBuffer) -> Tuple[int]:
        pass

    @abstractmethod
    def get_matrix_block(self, buffer: PETAceBuffer, row_start: int, col_start: int, row_num: int,
                         col_num: int) -> PETAceBuffer:
        pass
