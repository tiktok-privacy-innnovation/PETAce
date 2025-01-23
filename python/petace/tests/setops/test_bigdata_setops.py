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
from petace.setops.bigdata_setops import BigdataSetOps
from petace.setops.pysetops import PSIScheme
from petace.tests.utils import TwoPartyTestBase
from crypto_data_engine import Context
from crypto_data_engine import DataType


class TestBigdataSetops(TwoPartyTestBase):

    def test_ecdh_psi(self, party):
        dir = os.path.dirname(os.path.abspath(__file__))
        engine_ctx_params = {"work_directory": "./test_data", 'engine_type': "local"}
        parties = {
            '0': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 21000,
                "local_port_base": 22000
            },
            '1': {
                "remote_addr": "127.0.0.1",
                "remote_port_base": 22000,
                "local_port_base": 21000
            }
        }
        ctx = Context(**engine_ctx_params)
        setops = BigdataSetOps(party, ctx, "socket", parties)
        if party == 0:
            input_path = dir + "/test_data/dataset1.csv"
            input = ctx.csv_dataset(file_path=input_path, data_type=[DataType.String])
        else:
            input_path = dir + "/test_data/dataset2.csv"
            input = ctx.csv_dataset(file_path=input_path, data_type=[DataType.String])
        if party == 0:
            output_path = dir + "/test_data/dataset1_output.csv"
            output = ctx.csv_dataset(file_path=output_path)
        else:
            output_path = dir + "/test_data/dataset2_output.csv"
            output = ctx.csv_dataset(file_path=output_path)
        setops.run([input, output], True, setops_scheme=PSIScheme.ECDH_PSI, sort_result=True)
        assert output.shape()[0] == 3
