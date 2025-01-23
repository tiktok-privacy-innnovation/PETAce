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

import petace.securenumpy as snp
from petace.tests.utils import SnpBigdataTestBase


class TestAdd(SnpBigdataTestBase):

    def test_add_plain(self, party_id):
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64).reshape((3, 4))
        c0 = snp.array(array, 0)
        c1 = c0 + c0
        res2_plain = c1.reveal_to(0)
        if party_id == 0:
            print(res2_plain)

    def test_add_plain1(self, party_id):
        dir = os.path.dirname(os.path.abspath(__file__))
        a_path = dir + "/test_data/a.csv"
        c0 = snp.array(a_path, 0)
        c1 = c0 + c0
        self.res2_plain = c1.reveal_to(0)
        if party_id == 0:
            print(self.res2_plain)
