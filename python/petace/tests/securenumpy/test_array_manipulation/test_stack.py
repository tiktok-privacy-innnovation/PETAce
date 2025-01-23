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
class TestVstack(SnpTestBase):

    def test_0d(self, party_id):
        a = np.array(1)
        b = np.array(2)
        a_cipher = snp.array(a, 0)
        b_cipher = snp.array(b, 0)
        res = snp.vstack([a_cipher, b_cipher])
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res_plain, np.vstack([a, b]))

    def test_1d_2d(self, party_id):
        np.random.seed(43)
        for shape in ((10,), (10, 20)):
            arrays = [np.random.random(shape) for _ in range(5)]
            arrays_cipher = [snp.array(arr, 0) for arr in arrays]
            res_cipher = snp.vstack(arrays_cipher)
            npt.assert_equal(res_cipher.shape, np.vstack(arrays).shape)
            res = res_cipher.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res, np.vstack(arrays), decimal=4)

    def test_non_iterable(self, _):
        a_cipher = snp.array(np.array(1), 0)
        with npt.assert_raises(TypeError):
            snp.vstack(a_cipher)

    def test_empty_input(self, _):
        with npt.assert_raises(ValueError):
            snp.vstack([])


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestHstack(SnpTestBase):

    def test_non_iterable(self, _):
        a_cipher = snp.array(np.array(1), 0)
        with npt.assert_raises(TypeError):
            snp.hstack(a_cipher)

    def test_empty_input(self, _):
        with npt.assert_raises(ValueError):
            snp.hstack([])

    def test_0d(self, party_id):
        a = np.array(1)
        b = np.array(2)
        a_cipher = snp.array(a, 0)
        b_cipher = snp.array(b, 0)
        res = snp.hstack([a_cipher, b_cipher])
        res_plain = res.reveal_to(0)
        if party_id == 0:
            npt.assert_equal(res_plain, np.hstack([a, b]))

    def test_1d_2d(self, party_id):
        np.random.seed(43)
        for shape in ((10,), (10, 20)):
            arrays = [np.random.random(shape) for _ in range(2)]
            arrays_cipher = [snp.array(arr, 0) for arr in arrays]
            res_cipher = snp.hstack(arrays_cipher)
            npt.assert_equal(res_cipher.shape, np.hstack(arrays).shape)
            res = res_cipher.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res, np.hstack(arrays), decimal=4)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestColumnStack(SnpTestBase):

    def test_non_iterable(self, _):
        a_cipher = snp.array(np.array(1), 0)
        with npt.assert_raises(TypeError):
            snp.column_stack(a_cipher)

    def test_1d_2d(self, party_id):
        np.random.seed(43)
        for shape in (
            (3),
            (10, 20),
        ):
            arrays = [np.random.random(shape) for _ in range(2)]
            arrays_cipher = [snp.array(arr, 0) for arr in arrays]
            res_cipher = snp.column_stack(arrays_cipher)
            npt.assert_equal(res_cipher.shape, np.column_stack(arrays).shape)
            res = res_cipher.reveal_to(0)
            if party_id == 0:
                npt.assert_almost_equal(res, np.column_stack(arrays), decimal=4)


@pytest.mark.skipif(os.environ.get("PETACE_ENGINE_BACKEND", "duet") != PETAceBackendType.Duet,
                    reason="unsupported backend")
class TestRowStack(SnpTestBase):

    def test_empty_input(self, _):
        with npt.assert_raises(ValueError):
            snp.row_stack([])
