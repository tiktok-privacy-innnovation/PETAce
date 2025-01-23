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
class TestDot(SnpTestBase):

    def test_1d(self, party_id):
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        a_cipher = snp.array(a, 0)
        b_cipher = snp.array(b, 1)
        c_cipher = snp.dot(a_cipher, b_cipher)
        c_plain = c_cipher.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(c_plain, np.dot(a, b), decimal=4)

    def test_2d(self, party_id):
        np.random.seed(43)
        a = np.random.random((3, 2))
        b = np.random.random((2, 3))
        a_cipher = snp.array(a, 0)
        b_cipher = snp.array(b, 1)
        c_cipher = snp.dot(a_cipher, b_cipher)
        c_plain = c_cipher.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(c_plain, np.dot(a, b), decimal=4)
