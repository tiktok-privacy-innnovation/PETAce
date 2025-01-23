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
import operator

import pytest
import numpy as np
import numpy.testing as npt

import petace.securenumpy as snp
from petace.backend import PETAceBackendType
from petace.tests.utils import SnpTestBase


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestLogicalOp(SnpTestBase):
    ops = {operator.and_, operator.or_, operator.xor}

    def test_plain(self, party_id):
        p0 = np.array([True, True, False])
        p1 = np.array([True, False, True])
        c0 = snp.array(p0, 0, dtype=np.bool_)
        for op in self.ops:
            res1 = op(c0, p1)
            res2 = op(p1, c0)
            res1_plain = res1.reveal_to(0)
            res2_plain = res2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res1_plain, op(p0, p1))
                npt.assert_equal(res2_plain, op(p0, p1))

    def test_scalar(self, party_id):
        p0 = np.array([True, True, False])
        c0 = snp.array(p0, 0, dtype=np.bool_)
        scalar = False
        for op in self.ops:
            res1 = op(c0, scalar)
            res2 = op(scalar, c0)
            res1_plain = res1.reveal_to(0)
            res2_plain = res2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res1_plain, op(p0, scalar))
                npt.assert_equal(res2_plain, op(scalar, p0))

    def test_cipher(self, party_id):
        p0 = np.array([[True, True, False]])
        p1 = np.array([[True, False, True]])
        c0 = snp.array(p0, 0, dtype=np.bool_)
        c1 = snp.array(p1, 0, dtype=np.bool_)
        for op in self.ops:
            res = op(c0, c1)
            res_plain = res.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(res_plain, op(p0, p1))
