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
class TestSetItem(SnpTestBase):

    def test_1d(self, party_id):
        data = np.array([1, 2, 3, 4, 5])
        c1 = snp.array(data, 0)
        # int index
        c1[0] = 10
        c1[1] = np.array(20)
        c1[2] = snp.array(np.array(30), 0)
        p1 = c1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p1, np.array([10, 20, 30, 4, 5]))

        # slice index
        c2 = snp.array(data, 0)
        c2[1:3] = 10
        p2 = c2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p2, np.array([1, 10, 10, 4, 5]))
        c2[1:3] = np.array([20, 20])
        p2 = c2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p2, np.array([1, 20, 20, 4, 5]))
        c2[1:3] = snp.array(np.array([30, 30]), 0)
        p2 = c2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p2, np.array([1, 30, 30, 4, 5]))

    def test_raise(self, _):
        data = np.array([1, 2, 3, 4, 5])
        c1 = snp.array(data, 0)
        with npt.assert_raises(IndexError):
            c1[5] = 10
        with npt.assert_raises(ValueError):
            c1[1:2] = np.array([2, 3, 4, 5])

    def test_2d(self, party_id):
        data = np.array([[1, 2, 4, 5, 6], [7, 8, 9, 10, 11], [2, 3, 4, 5, 6]])
        c0 = snp.array(data, 0)
        # single index
        c0[0] = 1
        c0[1] = np.array([2, 3, 4, 5, 6])
        c0[1] = snp.array(np.array([7, 8, 9, 10, 11]), 0)
        p0 = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p0, np.array([[1, 1, 1, 1, 1], [7, 8, 9, 10, 11], [2, 3, 4, 5, 6]]))
        # slice index
        c1 = snp.array(data, 0)
        c1[:1] = 1
        c1[2:3] = np.array([2, 3, 4, 5, 16])
        c1[1:2] = snp.array(np.array([3, 3, 3, 3, 3]), 0)
        p1 = c1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p1, np.array([[1, 1, 1, 1, 1], [3, 3, 3, 3, 3], [2, 3, 4, 5, 16]]))

        # tuple index
        c2 = snp.array(data, 0)
        c2[0, 0] = 10
        c2[0, 1] = snp.array(np.array(20), 0)
        c2[1:, 2:-1] = snp.array(np.array([[12, 13], [17, 18]]), 0)
        p2 = c2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p2, np.array([[10, 20, 4, 5, 6], [7, 8, 12, 13, 11], [2, 3, 17, 18, 6]]))
