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

import pytest
import numpy as np
import numpy.testing as npt

import petace.securenumpy as snp
from petace.backend import PETAceBackendType
from petace.tests.utils import SnpTestBase


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestMatmul(SnpTestBase):

    def test_cipher(self, party_id):
        # 2d @ 2d
        np.random.seed(43)
        arr1 = snp.array(np.arange(10).reshape((2, 5)), 0)
        arr2 = snp.array(np.arange(10).reshape((5, 2)), 0)
        res = arr1 @ arr2
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape((2, 5)) @ np.arange(10).reshape((5, 2)))

        # 1d @ 1d
        c0 = snp.arange(5)
        c1 = snp.arange(5)
        res = c0 @ c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(5) @ np.arange(5))

        # 1d @ 2d
        c0 = snp.arange(5)
        c1 = snp.array(np.arange(10).reshape((5, 2)), 0)
        res = c0 @ c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(5) @ np.arange(10).reshape((5, 2)))

        # 2d @ 1d
        c0 = snp.array(np.arange(10).reshape((2, 5)), 0)
        c1 = snp.arange(5)
        res = c0 @ c1
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.arange(10).reshape((2, 5)) @ np.arange(5))

    def test_plain(self, party_id):
        # 2d @ 2d
        p0 = np.arange(10).reshape((2, 5))
        p1 = np.arange(10).reshape((5, 2))
        c0 = snp.array(p0, 0)
        res1 = p1 @ c0
        res2 = c0 @ p1
        res1_plain = res1.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, p1 @ p0)
            npt.assert_almost_equal(res2_plain, p0 @ p1)
        # 1d @ 1d
        p0 = np.arange(5)
        c0 = snp.array(p0, 0)
        res1 = p0 @ c0
        res2 = c0 @ p0
        res1_plain = res1.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, p0 @ p0)
            npt.assert_almost_equal(res2_plain, p0 @ p0)

        # 1d @ 2d
        p0 = np.arange(5)
        c0 = snp.array(p0, 0)
        p1 = np.arange(10).reshape((5, 2))
        c1 = snp.array(p1, 0)
        res1 = p0 @ c1
        res2 = c0 @ p1
        res1_plain = res1.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, p0 @ p1)
            npt.assert_almost_equal(res2_plain, p0 @ p1)

        # 2d @ 1d
        p0 = np.arange(10).reshape((2, 5))
        c0 = snp.array(p0, 0)
        p1 = np.arange(5)
        c1 = snp.array(p1, 0)
        res1 = p0 @ c1
        res2 = c0 @ p1
        res1_plain = res1.reveal_to(0)
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, p0 @ p1)
            npt.assert_almost_equal(res2_plain, p0 @ p1)
