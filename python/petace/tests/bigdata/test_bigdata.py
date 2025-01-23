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
import numpy as np
import pandas as pd
from petace.tests.utils import TwoPartyTestBase
from petace.backend.bigdata import BigDataVM


class TestBigData(TwoPartyTestBase):

    def test_map_case_0(self, party):
        test_op = {"add", "sub", "mul", "div"}
        for op in test_op:
            dir = os.path.dirname(os.path.abspath(__file__))
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

            vm = BigDataVM(party, {"work_directory": dir + "/test_data"}, 'socket', parties, 1)
            pa = vm.new_private_double_matrix(0)
            if party == 0:
                input_array = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
                vm.set_private_double_matrix(input_array, pa)

            sa = vm.new_airth_matrix()
            sb = vm.new_airth_matrix()
            pb = vm.new_private_double_matrix(0)
            data_to_share = [pa, sa]
            vm.exec_code(["share", *[obj.dtype for obj in data_to_share]], data_to_share)
            data_to_map = [sa, sa, sb]
            vm.exec_code([op, *[obj.dtype for obj in data_to_map]], data_to_map)
            data_to_reveal = [sb, pb]
            vm.exec_code(["reveal", *[obj.dtype for obj in data_to_reveal]], data_to_reveal)
            vm.run()
            if party == 0:
                real_ret_array = vm.get_private_double_matrix(pb)
                if op == "add":
                    assert np.allclose(input_array + input_array, real_ret_array, atol=0.001)
                elif op == "sub":
                    assert np.allclose(input_array - input_array, real_ret_array, atol=0.001)
                elif op == "mul":
                    assert np.allclose(input_array * input_array, real_ret_array, atol=0.001)
                elif op == "div":
                    assert np.allclose(input_array / input_array, real_ret_array, atol=0.01)
            vm.engine_ctx.close()

    def test_agg_case_0(self, party):
        test_op = {"groupby_max", "groupby_min", "groupby_sum", "groupby_count"}
        for op in test_op:
            dir = os.path.dirname(os.path.abspath(__file__))
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

            vm = BigDataVM(party, {"work_directory": dir + "/test_data"}, 'socket', parties, 5)
            # pylint: disable=too-many-try-statements
            try:
                pdata = vm.new_private_double_matrix(1)
                pgroup = vm.new_private_double_matrix(0)
                ponehot = vm.new_private_double_matrix(0)
                group_data = np.array([1, 2, 3, 4, 1, 2, 3, 9, 1, 3]).reshape(10, 1).astype(np.float64)
                data = np.array([1, 2, 3, 4, 5, 6, 0, 12, 15, 123]).reshape(10, 1).astype(np.float64)
                if party == 0:
                    vm.set_private_double_matrix(group_data, pgroup)
                else:
                    vm.set_private_double_matrix(data, pdata)
                sonehot = vm.new_airth_matrix()
                sdata = vm.new_airth_matrix()
                sret = vm.new_airth_matrix()
                data_to_onehot_encode = [pgroup, ponehot]
                vm.exec_code(["onehot_encode", *[obj.dtype for obj in data_to_onehot_encode]], [pgroup, ponehot])
                data_to_share_onehot = [ponehot, sonehot]
                vm.exec_code(["share", *[obj.dtype for obj in data_to_share_onehot]], [ponehot, sonehot])
                data_to_share = [pdata, sdata]
                vm.exec_code(["share", *[obj.dtype for obj in data_to_share]], [pdata, sdata])
                data_to_groupby_max = [sdata, sonehot, sret]
                vm.exec_code([op, *[obj.dtype for obj in data_to_groupby_max]], [sdata, sonehot, sret])
                pret = vm.new_private_double_matrix(0)
                data_to_reveal = [sret, pret]
                vm.exec_code(["reveal", *[obj.dtype for obj in data_to_reveal]], [sret, pret])
                vm.run()
                if party == 0:
                    real_ret_array = vm.get_private_double_matrix(pret)
                    real_ret_array = real_ret_array.reshape(1, -1)
                    combined = np.hstack((group_data, data))
                    df = pd.DataFrame(combined, columns=['group_data', 'data'])
                    if op == "groupby_max":
                        assert np.allclose(df.groupby('group_data')['data'].max().values, real_ret_array, atol=0.1)
                    elif op == "groupby_min":
                        assert np.allclose(df.groupby('group_data')['data'].min().values, real_ret_array, atol=0.1)
                    elif op == "groupby_sum":
                        assert np.allclose(df.groupby('group_data')['data'].sum().values, real_ret_array, atol=0.1)
                    elif op == "groupby_count":
                        assert np.allclose(df.groupby('group_data')['data'].count().values, real_ret_array, atol=0.1)
            except Exception as e:
                print(e)
                raise e
            finally:
                vm.engine_ctx.close()
