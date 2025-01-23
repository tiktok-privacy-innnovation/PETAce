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
class TestArgmaxMax(SnpTestBase):

    def test_2d(self, party_id):
        np.random.seed(43)
        data_a = np.random.random((2, 5))
        for axis in (0, 1, None):
            p0 = snp.array(data_a, 0)
            a1, a2 = snp.argmax_and_max(p0, axis=axis)
            p1 = a1.reveal_to(0)
            p2 = a2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(p1, np.argmax(data_a, axis=axis))
                npt.assert_almost_equal(p2, np.max(data_a, axis=axis), decimal=5)

    def test_1d(self, party_id):
        np.random.seed(43)
        data_a = np.random.random(5)
        p0 = snp.array(data_a, 0)
        a1, a2 = snp.argmax_and_max(p0)
        p1 = a1.reveal_to(0)
        p2 = a2.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(p1, np.argmax(data_a))
            npt.assert_almost_equal(p2, np.max(data_a), decimal=5)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestArgminMin(SnpTestBase):

    def test_2d(self, party_id):
        np.random.seed(43)
        data_a = np.random.random((2, 5))
        p0 = snp.array(data_a, 0)
        for axis in (0, 1, None):
            a1, a2 = snp.argmin_and_min(p0, axis=axis)
            p1 = a1.reveal_to(0)
            p2 = a2.reveal_to(0)
            if party_id == 0:
                npt.assert_equal(p1, np.argmin(data_a, axis=axis))
                npt.assert_almost_equal(p2, np.min(data_a, axis=axis), decimal=5)

    def test_1d(self, party_id):
        np.random.seed(43)
        data_a = np.random.random(5)
        p0 = snp.array(data_a, 0)
        a1, a2 = snp.argmin_and_min(p0)
        p1 = a1.reveal_to(0)
        p2 = a2.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(p1, np.argmin(data_a))
            npt.assert_almost_equal(p2, np.min(data_a), decimal=5)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestArgmax(SnpTestBase):

    def test_basic(self, party_id):
        np.random.seed(43)
        data_a = np.random.random((2, 5))
        p0 = snp.array(data_a, 0)
        a1 = snp.argmax(p0, axis=0)
        p1 = a1.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(p1, np.argmax(data_a, axis=0))


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestArgmin(SnpTestBase):

    def test_basic(self, party_id):
        np.random.seed(43)
        data_a = np.random.random((2, 5))
        p0 = snp.array(data_a, 0)
        a1 = snp.argmin(p0, axis=0)
        p1 = a1.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(p1, np.argmin(data_a, axis=0))
