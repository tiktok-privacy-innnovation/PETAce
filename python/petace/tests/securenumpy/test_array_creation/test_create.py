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
class TestEmpty(SnpTestBase):

    def test_basic(self, party_id):
        for shape in ((10,), (10, 20)):
            data = snp.empty(shape)
            npt.assert_equal(data.shape, shape)
            data_plian = data.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(data_plian.shape, shape)


class TestIdentity(SnpTestBase):

    def test_basic(self, party_id):
        data = snp.identity(10)
        npt.assert_equal(data.shape, (10, 10))
        data_plian = data.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(data_plian, np.identity(10))


class TestOnes(SnpTestBase):

    def test_basic(self, party_id):
        for shape in ((10,), (10, 20)):
            data = snp.ones(shape)
            npt.assert_equal(data.shape, shape)
            data_plian = data.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(data_plian, np.ones(shape))


class TestZeros(SnpTestBase):

    def test_basic(self, party_id):
        for shape in ((10,), (10, 20)):
            data = snp.zeros(shape)
            npt.assert_equal(data.shape, shape)
            data_plian = data.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(data_plian, np.zeros(shape))


class TestFull(SnpTestBase):

    def test_basic(self, party_id):
        value = 3.0
        for shape in ((10,), (10, 20)):
            data = snp.full(shape, value)
            npt.assert_equal(data.shape, shape)
            data_plian = data.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(data_plian, np.full(shape, value))


class TestCopy(SnpTestBase):

    def test_basic(self, party_id):
        arr = snp.ones((2, 3))
        arr2 = snp.copy(arr)
        plain1 = arr.reveal_to(0)
        plain2 = arr2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(plain1, plain2, decimal=8)
