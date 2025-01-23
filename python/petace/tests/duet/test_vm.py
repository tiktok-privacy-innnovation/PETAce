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

import petace.securenumpy as snp
from petace.tests.utils import SnpTestBase


class TestVM(SnpTestBase):

    def test_send_recv(self, party_id):
        vm = snp.get_engine().vm
        if party_id == 0:
            data = bytearray(b"Hello, World!")
            vm.send_buffer(data)
        else:
            data = vm.recv_buffer(13)
            assert bytes(data) == b"Hello, World!"

    def test_send_shape(self, party_id):
        shapes = (
            (),
            (1000,),
            (100000000, 1000),
        )
        vm = snp.get_engine()
        for shape in shapes:
            if party_id == 0:
                vm.send_shape(shape)
            else:
                recv_shape = vm.recv_shape()
                assert recv_shape == shape
