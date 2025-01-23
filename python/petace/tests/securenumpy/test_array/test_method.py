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
class TestReshape(SnpTestBase):

    def test_2d(self, party_id):
        a1 = np.arange(6).reshape((3, 2))
        a2 = np.arange(10, 16).reshape((3, 2))
        a1_cipher = snp.array(a1, 0)
        a2_cipher = snp.array(a2, 0)

        # 2-d new_shhape
        res1 = a1_cipher.reshape((2, 3)) + a2_cipher.reshape((2, 3))
        npt.assert_equal(res1.shape, (2, 3))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, a1.reshape((2, 3)) + a2.reshape((2, 3)))

        # 2-d new_shhape
        res1 = a1_cipher.reshape((2, -1)) + a2_cipher.reshape((2, -1))
        npt.assert_equal(res1.shape, (2, 3))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, a1.reshape((2, -1)) + a2.reshape((2, -1)))

        # 1-d new_shape
        res2 = a1_cipher.reshape(6)
        npt.assert_equal(res2.shape, (6,))
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res2_plain, a1.reshape(6))
        # -1 new_shape
        res3 = a1_cipher.reshape(-1)
        npt.assert_equal(res3.shape, (6,))
        res3_plain = res3.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res3_plain, a1.reshape(-1))

    def test_raise(self, _):
        a1 = np.arange(6).reshape((3, 2))
        a1_cipher = snp.array(a1, 0)
        with npt.assert_raises(ValueError):
            a1_cipher.reshape((2, 3, 1))
        with npt.assert_raises(ValueError):
            a1_cipher.reshape((8,))

    def test_0d(self, _):
        a1_cipher = snp.array(np.array(1), 0)
        res1 = a1_cipher.reshape(1)
        npt.assert_equal(res1.shape, (1,))
        res2 = res1.reshape((1, 1))
        npt.assert_equal(res2.shape, (1, 1))

    def test_1d(self, _):
        a1_cipher = snp.array(np.array([1, 2, 3]), 0)
        res1 = a1_cipher.reshape(-1)
        npt.assert_equal(res1.shape, (3,))
        res2 = res1.reshape((1, -1))
        npt.assert_equal(res2.shape, (1, 3))


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestFlatten(SnpTestBase):

    def test_basic(self, party_id):
        a1 = np.arange(6).reshape((3, 2))
        a1_cipher = snp.array(a1, 0)
        res = a1_cipher.flatten()
        npt.assert_equal(res.ndim, 1)
        npt.assert_equal(res.shape, (6,))
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res_plain, a1.flatten())


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestTranspose(SnpTestBase):

    def test_2d(self, party_id):
        a1 = np.arange(6).reshape((3, 2))
        a1_cipher = snp.array(a1, 0)

        res1 = a1_cipher.transpose()
        npt.assert_equal(res1.shape, (2, 3))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, a1.transpose())

    def test_0d(self, party_id):
        a1 = np.array(2)
        a1_cipher = snp.array(a1, 0)

        res1 = a1_cipher.transpose()
        npt.assert_equal(res1.shape, ())
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, a1.transpose())

    def test_1d(self, party_id):
        a1 = np.array([1, 2, 3])
        a1_cipher = snp.array(a1, 0)
        res1 = a1_cipher.transpose()
        npt.assert_equal(res1.shape, (3,))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_almost_equal(res1_plain, a1.transpose())


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestResize(SnpTestBase):

    def test_copies(self, party_id):
        arr = np.array([[1, 2], [3, 4]])
        arr_cipher = snp.array(arr, 0)
        res1 = arr_cipher.resize((2, 4))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res1_plain, np.resize(arr, (2, 4)))

        arr2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        arr_cipher2 = snp.array(arr2, 0)
        res2 = arr_cipher2.resize((4, 2))
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res2_plain, np.resize(arr2, (4, 2)))

        arr3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        arr_cipher3 = snp.array(arr3, 0)
        res3 = arr_cipher3.resize((4, 3))
        res3_plain = res3.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res3_plain, np.resize(arr3, (4, 3)))

        res4 = arr_cipher3.resize(12)
        res4_plain = res4.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res4_plain, np.resize(arr3, 12))

    def test_repeats(self, party_id):
        arr = np.array([1, 2, 3])
        arr_cipher = snp.array(arr, 0)
        res1 = arr_cipher.resize((2, 4))
        res1_plain = res1.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res1_plain, np.resize(arr, (2, 4)))

        arr2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        arr2_cipher = snp.array(arr2, 0)
        res2 = arr2_cipher.resize((4, 2))
        res2_plain = res2.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res2_plain, np.resize(arr2, (4, 2)))

        arr3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        arr3_cipher = snp.array(arr3, 0)
        res3 = arr3_cipher.resize((4, 3))
        res3_plain = res3.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res3_plain, np.resize(arr3, (4, 3)))

    def test_negative_resize(self, _):
        A = snp.arange(10)
        with npt.assert_raises(ValueError):
            A.resize((-10, -1))
        with npt.assert_raises(ValueError):
            A.resize(-1)

    def test_1d(self, party_id):
        arr = snp.arange(10)
        res = arr.resize((2, 4))
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res_plain, np.resize(np.arange(10), (2, 4)))
