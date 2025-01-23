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
class TestQuickSort(SnpTestBase):

    def test_case1(self, party_id):
        data_a = np.array([[1], [3], [5], [2]], dtype=np.float64)
        c0 = snp.array(data_a, 0)
        c0.quick_sort()
        p1 = c0.reveal_to(0)
        if party_id == 0:
            npt.assert_array_equal(p1.flatten(), np.sort(data_a.flatten()))

    def test_case2(self, party_id):
        np.random.seed(43)
        data = np.random.random((4, 5))
        data_cipher = snp.array(data, 0)
        data_cipher.quick_sort()
        data_plain = data_cipher.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(data_plain, np.sort(data, axis=None).reshape(data.shape), decimal=4)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestQuickSortByColumn(SnpTestBase):

    def test_case1(self, party_id):
        np.random.seed(43)
        data = np.random.random((4, 5))
        data_cipher = snp.array(data, 0)
        data_cipher_sort = data_cipher.quick_sort_by_column(0)
        data_plain = data_cipher_sort.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(data_plain, data[np.argsort(data[:, 0])], decimal=4)

    def test_case2(self, party_id):
        data_a = np.array([[1, 2], [3, 5], [5, 1], [2, 3]], dtype=np.float64)
        s0 = snp.array(data_a, 0)
        s1 = s0.quick_sort_by_column(0)
        s2 = s0.quick_sort_by_column(1)
        p1 = s1.reveal_to(0)
        p2 = s2.reveal_to(0)
        if party_id == 0:
            npt.assert_array_equal(p1, data_a[np.argsort(data_a[:, 0])])
            npt.assert_array_equal(p2, data_a[np.argsort(data_a[:, 1])])
