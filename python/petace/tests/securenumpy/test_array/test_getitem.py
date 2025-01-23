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
class TestGetItem(SnpTestBase):

    def test_1d(self, party_id):
        data_b = np.arange(18).astype(np.float64)
        p1 = snp.array(data_b, 0)
        test_cases = ((p1[1], data_b[1]), (p1[1:], data_b[1:]), (p1[-2:], data_b[-2:]), (p1[5:10], data_b[5:10]))
        for cipher, plain in test_cases:
            c0 = cipher.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(c0, plain)

    def test_2d(self, party_id):
        data_a = np.arange(18).reshape(9, 2).astype(np.float64)
        p0 = snp.array(data_a, 0)
        test_cases = ((p0[0], data_a[0]), (p0[5:], data_a[5:]), (p0[-5:], data_a[-5:]), (p0[1:9], data_a[1:9]),
                      (p0[-5:-1], data_a[-5:-1]), (p0[2, 1], data_a[2, 1]), (p0[1, :2], data_a[1, :2]),
                      (p0[1:, 1], data_a[1:, 1]), (p0[1:, 1:2], data_a[1:, 1:2]))
        for cipher, plain in test_cases:
            c0 = cipher.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(c0, plain)

    def test_1d_sum(self, _):
        data = snp.ones((10, 10))
        _ = data[:, 5] + data[5]
