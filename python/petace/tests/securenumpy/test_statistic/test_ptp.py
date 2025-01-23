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
class TestPtp(SnpTestBase):

    def test_axis0(self, party_id):
        np.random.seed(43)
        arr = np.random.random((2, 5))
        arr_cipher = snp.array(arr, 0)
        for axis in (0, 1, None):
            res = snp.ptp(arr_cipher, axis=axis)
            res_palin = res.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res_palin, np.ptp(arr, axis=axis), decimal=5)
