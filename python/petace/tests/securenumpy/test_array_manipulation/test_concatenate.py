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
class TestConcatenate(SnpTestBase):

    def test_none_axis(self, party_id):
        # 0d, 1d, 2d
        test_cases = (
            (np.array(1), np.array(2)),
            (np.array([1]), np.array(2)),
            (np.array([[1]]), np.array([[2]])),
        )
        for a, b in test_cases:
            a_cipher = snp.array(a, 0)
            b_cipher = snp.array(b, 0)
            res = snp.concatenate([a_cipher, b_cipher], axis=None)
            res_plain = res.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, np.concatenate([a, b], axis=None))

    def test_axis0(self, party_id):
        # 0d, 1d, 2d
        test_cases = (
            (np.array([1]), np.array([2])),
            (np.array([[1]]), np.array([[2]])),
        )
        for a, b in test_cases:
            a_cipher = snp.array(a, 0)
            b_cipher = snp.array(b, 0)
            res = snp.concatenate([a_cipher, b_cipher], axis=0)
            res_plain = res.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, np.concatenate([a, b], axis=0))

    def test_axis1(self, party_id):
        a = np.ones((2, 3))
        b = np.ones((2, 3))
        a_cipher = snp.array(a, 0)
        b_cipher = snp.array(b, 0)
        res = snp.concatenate([a_cipher, b_cipher], axis=1)
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res_plain, np.concatenate([a, b], axis=1))
