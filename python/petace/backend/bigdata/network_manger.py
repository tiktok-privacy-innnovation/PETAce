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

from enum import Enum
from typing import Dict
from petace.network import NetParams, NetScheme, NetFactory


class NetworkType(Enum):
    SOCKET = "socket"
    AGENT = "agent"


class NetworkManger:

    def __init__(self, type: NetworkType, params: Dict):
        """
        for socket:
        {
            "remote_addr": "127.0.0.1"
            "remote_port_base": 8890
            "local_port_base": 8891
        }
        """
        self.type = type
        self.params = params

    def get_network(self, count, index):
        net_scheme, net_params = None, NetParams()
        if self.type == NetworkType.SOCKET:
            net_scheme = NetScheme.SOCKET
            net_params.remote_addr = self.params["remote_addr"]
            net_params.remote_port = self.params["remote_port_base"] + (count * 7 % 1000) + index
            net_params.local_port = self.params["local_port_base"] + (count * 7 % 1000) + index
        elif self.type == NetworkType.AGENT:
            net_scheme = NetScheme.AGENT
            net_params.shared_topic = f"{self.params['shared_topic']}:{count}:{index}"
            net_params.local_agent = self.params["local_agent"]
            net_params.remote_party = self.params["remote_party"]
        else:
            raise ValueError(f"unknown network type {self.type}")
        return NetFactory.get_instance().build(net_scheme, net_params)
