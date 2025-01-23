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
class TestFromShare(SnpTestBase):

    def test_0d(self, party_id):
        np.random.seed(43)
        data = np.array(np.random.random())
        data_cipher = snp.array(data, 0)
        share = data_cipher.to_share()
        npt.assert_equal(share.shape, data.shape)
        data_cipher_new = snp.fromshare(share, np.float64)
        data_plain_new = data_cipher_new.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(data_plain_new, data, decimal=4)

    def test_1d(self, party_id):
        np.random.seed(43)
        data = np.random.random(100)
        data_cipher = snp.array(data, 0)
        share = data_cipher.to_share()
        npt.assert_equal(share.shape, data.shape)
        data_cipher_new = snp.fromshare(share, np.float64)
        data_plain_new = data_cipher_new.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(data_plain_new, data, decimal=4)

    def test_2d(self, party_id):
        np.random.seed(43)
        data = np.random.random((10, 20))
        data_cipher = snp.array(data, 0)
        share = data_cipher.to_share()
        npt.assert_equal(share.shape, data.shape)
        data_cipher_new = snp.fromshare(share, np.float64)
        data_plain_new = data_cipher_new.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(data_plain_new, data, decimal=4)
