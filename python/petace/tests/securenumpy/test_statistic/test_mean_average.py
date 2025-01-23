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
class TestMean(SnpTestBase):

    def test_basic(self, party_id):
        np.random.seed(43)
        arr = np.random.random((2, 5))
        arr_cipher = snp.array(arr, 0)
        for axis in (None, 0, 1):
            res = snp.mean(arr_cipher, axis=axis)
            res_palin = res.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res_palin, np.mean(arr, axis=axis), decimal=4)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestAverage(SnpTestBase):

    def test_1d(self, party_id):
        np.random.seed(43)
        arr = np.random.random(5)
        weight = np.random.random(5)
        arr_cipher = snp.array(arr, 0)
        res = snp.average(arr_cipher, weights=weight)
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.average(arr, weights=weight), decimal=4)

    def test_2d(self, party_id):
        np.random.seed(43)
        arr = np.random.random((2, 5))
        weight = np.random.random((2, 5))
        arr_cipher = snp.array(arr, 0)
        for axis in (None, 0, 1):
            res = snp.average(arr_cipher, axis=axis, weights=weight)
            res_plain = res.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res_plain, np.average(arr, axis=axis, weights=weight), decimal=3)

        # aixs = 1 and auto broadcast
        arr = np.random.random((2, 5))
        weight = np.random.random(5)
        weight_cipher = snp.array(weight, 0)
        arr_cipher = snp.array(arr, 0)
        res = snp.average(arr_cipher, axis=1, weights=weight_cipher)
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, np.average(arr, axis=1, weights=weight), decimal=3)
