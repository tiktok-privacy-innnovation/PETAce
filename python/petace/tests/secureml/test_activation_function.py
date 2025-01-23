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
import petace.secureml as sml

from petace.backend import PETAceBackendType
from petace.tests.utils import SnpTestBase


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestActivationFunction(SnpTestBase):

    def test_sigmoid(self, party_id):
        data_a = np.array([[-1.0], [-0.5], [0], [0.5], [1]], dtype=np.float64)
        real_data = 1. / (1 + np.exp(-data_a))
        p0 = snp.array(data_a, 0)
        s1 = sml.sigmoid(p0)
        p1 = s1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p1, real_data, decimal=3)

        def linear_sigmoid(array):
            output = array + 1 / 2
            output = np.where(output <= 1, output, np.ones(output.shape))
            output = np.where(output >= 0, output, np.zeros(output.shape))
            return output

        real_fitting_data = linear_sigmoid(data_a)
        s2 = sml.sigmoid(p0, 1)
        p2 = s2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(p2, real_fitting_data, decimal=5)
            npt.assert_allclose(p2, real_data, atol=0.4)
