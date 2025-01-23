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

import operator

import numpy as np
import numpy.testing as npt

import petace.securenumpy as snp
from petace.tests.utils import SnpTestBase


class TestCompOp(SnpTestBase):
    ops = {operator.lt, operator.gt, operator.eq, operator.ne, operator.ge, operator.le}

    def test_plain(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        p0 = np.random.random((2, 5))
        for op in self.ops:
            print(op.__name__)
            res = op(c0, p0)
            res2 = op(p0, c0)
            res_plain = res.reveal_to(0)
            res2_plain = res2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, op(np.arange(10).reshape(2, 5), p0))
                npt.assert_equal(res2_plain, op(p0, np.arange(10).reshape(2, 5)))

    def test_scalar(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        a = 1
        for op in self.ops:
            print(op.__name__)
            res = op(c0, a)
            res2 = op(a, c0)
            res_plain = res.reveal_to(0)
            res2_plain = res2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, op(np.arange(10).reshape(2, 5), a))
                npt.assert_equal(res2_plain, op(a, np.arange(10).reshape(2, 5)))

    def test_cipher(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        for op in self.ops:
            print(op.__name__)
            res = op(c0, c1)
            res_plain = res.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, op(np.arange(10), np.arange(10, 20)))
