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

import numpy as np
import numpy.testing as npt

import petace.securenumpy as snp
from petace.tests.utils import SnpTestBase


class TestReveal(SnpTestBase):

    def test_float64(self, party_id):
        data_a = np.array([[4, 4], [4, 4], [4, 4]], dtype=np.float64)

        p0 = snp.array(data_a, 0)
        p0_plain = p0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p0_plain, data_a)

    def test_bool(self, party_id):
        data_a = np.array([[True, False], [False, True]], dtype=np.bool_)

        p0 = snp.array(data_a, 0, dtype=np.bool_)
        p0_plain = p0.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(p0_plain, data_a)
