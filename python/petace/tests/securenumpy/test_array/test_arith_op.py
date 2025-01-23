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


class TestAdd(SnpTestBase):

    def test_add_plain(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        p0 = np.random.random((2, 5))
        res = c0 + p0
        res2 = p0 + c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, p0 + np.arange(10).reshape(2, 5), decimal=5)
            npt.assert_almost_equal(res2_plain, np.arange(10).reshape(2, 5) + p0, decimal=5)

    def test_add_scalar(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        res = c0 + 1
        res2 = 1 + c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape(2, 5) + 1, decimal=8)
            npt.assert_almost_equal(res2_plain, np.arange(10).reshape(2, 5) + 1, decimal=8)

    def test_add_cipher(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        res = c0 + c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) + np.arange(10, 20), decimal=8)

    def test_iadd(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        c0 += c1
        res_plain = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) + np.arange(10, 20), decimal=8)


class TestSub(SnpTestBase):

    def test_sub_plain(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        p0 = np.random.random((2, 5))
        res = c0 - p0
        res2 = p0 - c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape(2, 5) - p0, decimal=5)
            npt.assert_almost_equal(res2_plain, p0 - np.arange(10).reshape(2, 5), decimal=5)

    def test_sub_scalar(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        res = c0 - 1
        res2 = 1 - c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape(2, 5) - 1, decimal=8)
            npt.assert_almost_equal(res2_plain, 1 - np.arange(10).reshape(2, 5), decimal=8)

    def test_sub_cipher(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        res = c0 - c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) - np.arange(10, 20), decimal=8)

    def test_isub(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        c0 -= c1
        res_plain = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) - np.arange(10, 20), decimal=8)


class TestMul(SnpTestBase):

    def test_mul_plain(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        p0 = np.random.random((2, 5))
        res = c0 * p0
        res2 = p0 * c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, p0 * np.arange(10).reshape(2, 5), decimal=4)
            npt.assert_almost_equal(res2_plain, np.arange(10).reshape(2, 5) * p0, decimal=4)

    def test_sub_scalar(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        res = c0 * 1
        res2 = 1 * c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape(2, 5) * 1, decimal=8)
            npt.assert_almost_equal(res2_plain, np.arange(10).reshape(2, 5) * 1, decimal=8)

    def test_sub_cipher(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        res = c0 * c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) * np.arange(10, 20), decimal=8)

    def test_isub(self, party_id):
        c0 = snp.arange(10)
        c1 = snp.arange(10, 20)
        c0 *= c1
        res_plain = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10) * np.arange(10, 20), decimal=8)


class TestDiv(SnpTestBase):

    def test_div_plain(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(20, 30).reshape((2, 5)), 0)
        p0 = np.arange(10, 20).reshape((2, 5))
        res = c0 / p0
        res2 = p0 / c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(20, 30).reshape(2, 5) / p0, decimal=3)
            npt.assert_almost_equal(res2_plain, p0 / np.arange(20, 30).reshape(2, 5), decimal=3)

    def test_div_scalar(self, party_id):
        np.random.seed(43)
        c0 = snp.array(np.arange(10, 20).reshape((2, 5)), 0)
        res = c0 / 1
        res2 = 1 / c0
        res_plain = res.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10, 20).reshape(2, 5), decimal=3)
            npt.assert_almost_equal(res2_plain, 1. / np.arange(10, 20).reshape(2, 5), decimal=3)

    def test_div_cipher(self, party_id):
        c0 = snp.arange(20, 30)
        c1 = snp.arange(10, 20)
        res = c0 / c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(20, 30) / np.arange(10, 20), decimal=3)

    def test_idiv(self, party_id):
        c0 = snp.arange(20, 30)
        c1 = snp.arange(10, 20)
        c0 /= c1
        res_plain = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(20, 30) / np.arange(10, 20), decimal=3)
