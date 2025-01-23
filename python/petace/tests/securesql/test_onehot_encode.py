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

import petace.securenumpy as snp
import petace.securesql as ssql
from petace.tests.utils import SnpBigdataTestBase
from petace.backend.bigdata import BigDataVM
from petace.engine import PETAceEngine


class TestOnehot(SnpBigdataTestBase):

    def test_onehot_encode(self, party_id):
        dir = os.path.dirname(os.path.abspath(__file__))
        a_path = dir + "/../bigdata/test_data/one_hot.csv"
        parties = {
            '0': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 9000,
                "local_port_base": 8000
            },
            '1': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 8000,
                "local_port_base": 9000
            }
        }

        vm = BigDataVM(party_id, {"work_directory": dir + "/../bigdata/test_data"}, 'socket', parties, 5)
        engine = PETAceEngine(vm)
        snp.set_engine(engine)
        a_dataset = vm.new_csv_dataset(a_path, 5)
        ret = ssql.onehot_encode(a_dataset, 0)
        print(ret.file_path)
        vm.run()
        vm.close()

    def test_unique(self, party_id):
        dir = os.path.dirname(os.path.abspath(__file__))
        a_path = dir + "/../bigdata/test_data/one_hot.csv"
        parties = {
            '0': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 9000,
                "local_port_base": 8000
            },
            '1': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 8000,
                "local_port_base": 9000
            }
        }

        vm = BigDataVM(party_id, {"work_directory": dir + "/../bigdata/test_data"}, 'socket', parties, 5)
        engine = PETAceEngine(vm)
        snp.set_engine(engine)
        a_dataset = vm.new_csv_dataset(a_path, 5)
        ret = ssql.unique(a_dataset, 0)
        print(ret.file_path)
        vm.run()
        vm.close()
