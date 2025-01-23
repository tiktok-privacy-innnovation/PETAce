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

import os
import numpy as np
from petace.tests.utils import SinglePartyTestBase
from petace.backend.bigdata.crypto_data import Private, Shared, Public
from petace.backend.bigdata import BigDataPrivateType, BigDataShareType, BigDataPublicType

from crypto_data_engine import Context


class TestCryptoData(SinglePartyTestBase):

    def test_crypto_data(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        ctx = Context(work_directory=dir + "/test_data")
        data_a = ctx.csv_dataset()
        pa = Private(0, data_a, BigDataPrivateType.DOUBLE)
        pa.set_array(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64), index=False, header=False)
        assert pa.shape == (2, 3)
        data_b = ctx.csv_dataset()
        sb = Shared(data_b, BigDataShareType.DOUBLE)
        sb.set_array(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int64), index=False, header=False)
        assert sb.shape == (2, 3)
        data_c = ctx.csv_dataset()
        sc = Public(data_c, BigDataPublicType.DOUBLE)
        sc.set_array(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64), index=False, header=False)
        assert sc.shape == (2, 3)
        ctx.close()
