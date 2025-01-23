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

from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from crypto_data_engine import CSVDataset
from crypto_data_engine.data.scalar_dataset import ScalarDataset
from ._type import BigDataPrivateType, BigDataPublicType, BigDataShareType


class CryptoDataBase(ABC):

    def __init__(self, party: int, data, dtype):
        self._party = party
        self._data = data
        self._dtype = dtype

    @property
    def party(self):
        return self._party

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, CSVDataset):
            raise ValueError("must be CSVDataset")
        self._data = value

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._data.shape()

    @staticmethod
    def np2csv(data, file_path, *args, **kwargs):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            data.to_csv(file_path, *args, **kwargs)

    def set_array(self, npdata: np.ndarray, *args, **kwargs):
        if isinstance(self.data, CSVDataset):
            self.check_array_type(npdata)
            CryptoDataBase.np2csv(npdata, self.data.file_path, *args, **kwargs)
        else:
            raise TypeError(f"Data type should be CSVDataset, but given {type(self.data)}")

    def get_array(self, *args, **kwargs):
        if isinstance(self.data, CSVDataset):
            return pd.read_csv(self.data.file_path, *args, **kwargs).values
        raise TypeError(f"Data type should be CSVDataset, but given {type(self.data)}")

    def set_scalar(self, scala):
        if isinstance(self.data, ScalarDataset):
            self.data.value = scala
        else:
            raise TypeError(f"Data type should be ScalarDataset, but given {type(self.data)}")

    def get_scalar(self):
        if isinstance(self.data, ScalarDataset):
            return self.data.value
        raise TypeError(f"Data type should be ScalarDataset, but given {type(self.data)}")

    @abstractmethod
    def check_array_type(self, npdata: np.ndarray):
        pass


class Private(CryptoDataBase):

    def __init__(self, party: int, data: "CSVDataset", dtype: BigDataPrivateType = None):
        if party not in (0, 1):
            raise ValueError("Party must be 0 or 1.")
        super().__init__(party, data, dtype)

    def check_array_type(self, npdata: np.ndarray):
        if self._dtype is None:
            raise TypeError("Data type shouldn't be None.")

        if self._dtype == BigDataPrivateType.DOUBLE and npdata.dtype != np.float64:
            raise TypeError(f"Need np.float64, but given {npdata.dtype}")

        if self._dtype == BigDataPrivateType.BOOL and npdata.dtype != np.bool_:
            raise TypeError(f"Need np.bool_, but given {npdata.dtype}")
        return True


class Shared(CryptoDataBase):

    def __init__(self, data: "CSVDataset", dtype: BigDataShareType = None):
        super().__init__(-1, data, dtype)

    def check_array_type(self, npdata: np.ndarray):
        if self._dtype is None:
            raise TypeError("Data type shouldn't be None.")
        if npdata.dtype != np.int64:
            raise TypeError(f"Need np.int64, but given {npdata.dtype}")
        return True


class Public(CryptoDataBase):

    def __init__(self, data: Union["CSVDataset", "ScalarDataset"], dtype: BigDataPublicType = None):
        super().__init__(2, data, dtype)

    def check_array_type(self, npdata: np.ndarray):
        if self._dtype is None:
            raise TypeError("Data type shouldn't be None.")

        if self._dtype == BigDataPublicType.DOUBLE and npdata.dtype != np.float64:
            raise TypeError(f"Need np.float64, but given {npdata.dtype}")
        if self._dtype == BigDataPublicType.BOOL and npdata.dtype != np.bool_:
            raise TypeError(f"Need np.bool_, but given {npdata.dtype}")
        return True
