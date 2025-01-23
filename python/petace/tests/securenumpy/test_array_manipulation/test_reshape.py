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
import numpy.testing as npt

import petace.securenumpy as snp
from petace.backend import PETAceBackendType
from petace.tests.utils import SnpTestBase


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestReshape(SnpTestBase):
    """more test cases can be found in test_array/test_methods.py
    """

    def test_basic(self, _):
        a = snp.arange(10)
        a1 = snp.reshape(a, (2, 5))
        npt.assert_equal(a1.shape, (2, 5))


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestResize(SnpTestBase):
    """more test cases can be found in test_array/test_methods.py
    """

    def test_basic(self, _):
        a = snp.arange(10)
        a1 = snp.resize(a, (2, 6))
        npt.assert_equal(a1.shape, (2, 6))
