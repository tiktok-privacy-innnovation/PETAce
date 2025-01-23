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
class TestSort(SnpTestBase):

    def test_basic(self, party_id):
        np.random.seed(43)
        p0 = np.random.random((2, 5))
        c0 = snp.array(p0, 0)
        cipher_res = snp.sort(c0, axis=None)
        res = cipher_res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res, np.sort(p0, axis=None), decimal=5)

    def test_1d(self, party_id):
        np.random.seed(43)
        p0 = np.random.random(5)
        c0 = snp.array(p0, 0)
        cipher_res = snp.sort(c0)
        res = cipher_res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res, np.sort(p0), decimal=5)
