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

from ..utils import NetworkTestBase


class TestNetwork(NetworkTestBase):

    def test_network(self, party):
        data = b"hello world"
        if party == 0:
            self.net.send_data(data, len(data))
        else:
            ret = bytearray(len(data))
            self.net.recv_data(ret, len(data))
            print(ret)
            assert bytes(ret) == data

    # def test_agent(self, party):
    #     print(NetScheme.AGENT)
    #     net_params = NetParams()
    #     if party == 0:
    #         net_params.shared_topic = "j_20240601180000_1.psi"
    #         net_params.remote_party = "party_b"
    #         net_params.local_agent = "127.0.0.1:1235"
    #     else:
    #         net_params.shared_topic = "j_20240601180000_1.psi"
    #         net_params.remote_party = "party_a"
    #         net_params.local_agent = "127.0.0.1:1235"
    #     net = NetFactory.get_instance().build(NetScheme.AGENT, net_params)
    #     data = b"hello world"
    #     if party == 0:
    #         net.send_data(data, len(data))
    #     else:
    #         ret = bytearray(len(data))
    #         net.recv_data(ret, len(data))
    #         print(ret)
    #         assert bytes(ret) == data
