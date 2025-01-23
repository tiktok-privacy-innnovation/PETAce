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
import math
import os
import random
import string
import hashlib
from typing import List, Union

from crypto_data_engine import Context, Partition, DatasetBase

from petace.backend.bigdata.network_manger import NetworkManger, NetworkType
from petace.setops.pjc import PJC
from petace.setops.psi import PSI
from petace.setops.pysetops import PSIScheme, PJCScheme


class BigdataSetOps:
    """
    Class for performing large-scale private set intersection (PSI) and private join and compute(PJC).

    The overall workflow is as follows:
        1. Sample a random hash function h() as a partitioner in bigdata storage.
        2. Assign items to buckets according to the hash function h(): x will be assigned to the h(x)-th bucket.
        3. DataSet A (resp. DataSet B) is partitioned into m smaller datasets A_1, A_2, ..., A_m (resp. B_1, B_2, ..., B_m).
        4. Compute dummy ratio delta as programmed in SetOpsBucketParameters._decide_bucket_parameters.
        5. For all i from 1 to m, pad dataset A_i and B_i with randomly sampled dummy items.Denote the padded datasets as A'_i and B'_i.
        6. For all buckets run in-memory PSI/PJC on A'_i and B'_i.
        7. Merge the intersections from all in-memory PSI/PJC instances.
    In terms of the performance of PSI and PJC, the recommended number of items in each bucket is 1,000,000.

    Parameters:
    ----------
    party : int
        The party id.
    engine_ctx : object
        The engine context.
    net_scheme : str
        The network scheme, agent or socket.
    net_params : dict
        The params of network.

    Attributes:
    ----------
    party : int
        The party id.
    engine_ctx : object
        The engine context.
    count : int
        The count of PSI instances.
    net : Network
        The network to agree on parameters.
    item_num_per_bucket : int
        The default number of items in each bucket.
    """

    def __init__(self, party: int, engine_ctx: "Context", net_scheme, net_params):
        if party not in (0, 1):
            raise ValueError("Invalid party id")
        self.party = party
        self.engine_ctx = engine_ctx
        self.count = 0
        self.net = NetworkManger(NetworkType.SOCKET if net_scheme == 'socket' else NetworkType.AGENT,
                                 net_params[str(party)]).get_network(0, -1)
        self.kwargs = {'scheme': net_scheme, 'party': str(party), 'parties': net_params, 'vm_mode': 'none'}
        self.item_num_per_bucket = 1000000

    @staticmethod
    def exec_setops_by_partition(_, inputs: List["Partition"], outputs: List["Partition"], **kwargs) -> bool:
        # pylint: disable=too-many-statements
        party = int(kwargs['party'])
        count = kwargs["count"]
        maximum_data_size = kwargs["maximum_data_size"]
        item_num_per_bucket = kwargs["item_num_per_bucket"]
        statistical_security_parameter = kwargs["statistical_security_parameter"]
        obtain_result = kwargs["obtain_result"]
        setops_scheme_str = kwargs["setops_scheme_str"]
        sort_result = kwargs["sort_result"]

        schemes = {"ecdh": PSIScheme.ECDH_PSI, "kkrt": PSIScheme.KKRT_PSI, "circuit": PJCScheme.CIRCUIT_PSI}
        if setops_scheme_str not in schemes:
            raise ValueError(f"Invalid setops scheme {setops_scheme_str}")
        setops_scheme = schemes[setops_scheme_str]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Invalid inputs or outputs")
        index = inputs[0]._index
        if index < 0:
            raise ValueError("Invalid index")
        if party not in (0, 1):
            raise ValueError("Invalid party id")

        net_scheme = NetworkType.SOCKET if kwargs.get("net_scheme", 'socket') == 'socket' else NetworkType.AGENT
        net_params = kwargs['parties'][str(party)]
        net = NetworkManger(net_scheme, net_params).get_network(count, index)
        if setops_scheme in (PSIScheme.ECDH_PSI, PSIScheme.KKRT_PSI):
            memory_setops = PSI(net, party, setops_scheme)
        elif setops_scheme == PJCScheme.CIRCUIT_PSI:
            memory_setops = PJC(net, party, setops_scheme)
        else:
            raise ValueError("Invalid setops scheme")

        plain_data = inputs[0].read()
        if len(plain_data) == 0:
            payload_num = 0
        else:
            payload_num = len(plain_data[0]) - 1

        setops_bucket_parameters = SetOpsBucketParameters(party, maximum_data_size, item_num_per_bucket,
                                                          statistical_security_parameter)
        if isinstance(memory_setops, PSI):
            dummies = setops_bucket_parameters.get_dummies(index, len(plain_data))
            setops_input = [item[0] for item in plain_data]
            setops_input.extend(dummies[0])
            result = memory_setops.process(setops_input, obtain_result, False)
            if sort_result:
                result.sort()
            result = [[row] for row in result]
        elif isinstance(memory_setops, PJC):
            dummies = setops_bucket_parameters.get_dummies(index, len(plain_data), payload_num)
            setops_input_key = [item[0] for item in plain_data]
            setops_input_features = [[item[i] for item in plain_data] for i in range(1, payload_num + 1)]
            setops_input_key.extend(dummies[0])
            for i in range(payload_num):
                setops_input_features[i].extend(dummies[i + 1])
            result = memory_setops.process(setops_input_key, setops_input_features, False)
            result = [[row[i] for row in result] for i in range(len(result[0]))]
        else:
            raise ValueError("Invalid setops scheme")
        outputs[0].save(result)
        return True

    def get_partitioner(self) -> "callable":
        """
        Get a partitioner key.
        """
        seed = os.urandom(16)
        remote_seed = bytes(16)
        if self.party == 0:
            self.net.recv_data(remote_seed, 16)
            self.net.send_data(seed, 16)
        else:
            self.net.send_data(seed, 16)
            self.net.recv_data(remote_seed, 16)
        key = bytes([b1 ^ b2 for b1, b2 in zip(seed, remote_seed)])
        return generate_partitioner(key)

    def get_maximum_data_size(self, data_size: int) -> int:
        """
        Get the maximum data size given two inputs.
        """
        remote_data_size = bytes(8)
        if self.party == 0:
            self.net.recv_data(remote_data_size, 8)
            self.net.send_data(data_size.to_bytes(8, 'big'), 8)
        else:
            self.net.send_data(data_size.to_bytes(8, 'big'), 8)
            self.net.recv_data(remote_data_size, 8)
        return max(int.from_bytes(remote_data_size, 'big'), data_size)

    def run(self,
            data: List["DatasetBase"],
            obtain_result: bool,
            setops_scheme: Union["PSIScheme", "PJCScheme"] = PSIScheme.ECDH_PSI,
            statistical_security_parameter: int = 80,
            sort_result: bool = False):
        """
        Run setops.
        """
        schemes = {PSIScheme.ECDH_PSI: "ecdh", PSIScheme.KKRT_PSI: "kkrt", PJCScheme.CIRCUIT_PSI: "circuit"}
        if len(data) > 2:
            raise ValueError("psi only support one input and one output")
        if setops_scheme not in schemes:
            raise ValueError(f"Invalid setops scheme {setops_scheme}")
        setops_scheme_str = schemes[setops_scheme]
        maximum_data_size = self.get_maximum_data_size(data[0].shape()[0])
        partitioner = self.get_partitioner()
        data[0].set_partitioner(partitioner)
        if maximum_data_size < self.item_num_per_bucket:
            item_num_per_bucket = 10**(len(str(abs(maximum_data_size))))
        else:
            item_num_per_bucket = self.item_num_per_bucket
        setops_bucket_parameters = SetOpsBucketParameters(self.party, maximum_data_size, item_num_per_bucket,
                                                          statistical_security_parameter)
        data[0].repartition(setops_bucket_parameters.get_bucket_num())

        self.kwargs.update({
            "count": self.count,
            "maximum_data_size": maximum_data_size,
            "item_num_per_bucket": item_num_per_bucket,
            "statistical_security_parameter": statistical_security_parameter,
            "obtain_result": obtain_result,
            "setops_scheme_str": setops_scheme_str,
            "sort_result": sort_result,
        })
        self.engine_ctx.compute(None, self.exec_setops_by_partition, [data[0]], [data[1]], **self.kwargs)
        self.engine_ctx.execute()
        self.engine_ctx.close()


class SetOpsBucketParameters:
    """
    SetOps buckter parameters.

    Parameters:
    ----------
    party : int
        The party id of the set owner.
    maximum_data_size : int
        The maximum size of the set.
    item_num_per_bucket : int
        The number of items in each bucket.
    statistical_security_parameter : int
        The statistical security parameter.

    Attributes:
    ----------
    party : int
        The party id of the set owner.
    maximum_data_size : int
        The maximum size of the set.
    item_num_per_bucket : int
        The number of items in each bucket.
    sigma : int
        The statistical security parameter.
    bucket_num : int
        The number of buckets.
    delta : float
        The dummy ratio.
    """

    def __init__(self,
                 party: int,
                 maximum_data_size: int,
                 item_num_per_bucket: int = 1000000,
                 statistical_security_parameter: int = 80) -> None:
        self.party = party
        self.maximum_data_size = maximum_data_size
        self.item_num_per_bucket = item_num_per_bucket
        self.sigma = statistical_security_parameter
        self._decide_bucket_parameters()

    def _decide_bucket_parameters(self):
        """
        Decide the bucket parameters.
        """
        self.bucket_num = (self.maximum_data_size + self.item_num_per_bucket - 1) // self.item_num_per_bucket
        m = self.bucket_num
        n = self.maximum_data_size
        self.delta = ((3 * m / n) * (self.sigma * math.log(2) + math.log(m)))**0.5

    def get_bucket_num(self) -> int:
        """
        Get the number of buckets.
        """
        return self.bucket_num

    def get_dummies(self, bucket_index: int, cur_item_num: int, payload_num: int = 0) -> List[List[str]]:
        """
        Get the dummies.
        """
        if bucket_index < 0 or bucket_index >= self.bucket_num:
            raise ValueError("Invalid bucket index")
        if cur_item_num < 0 or cur_item_num > self.item_num_per_bucket * (1 + self.delta):
            raise ValueError("Invalid item number")
        dummies_num = int(self.item_num_per_bucket * (1 + self.delta)) - cur_item_num
        suffix = f'party_{self.party}_bucket_{bucket_index}'
        dummies = [[
            ''.join(random.sample(string.ascii_letters + string.digits, 32)) + suffix for _ in range(dummies_num)
        ]]
        if payload_num > 0:
            for _ in range(payload_num):
                dummies.append([random.getrandbits(64) for _ in range(dummies_num)])
        return dummies


def generate_partitioner(key: bytes = b''):
    """
        Get a partitioner. The dataset will be partitioned into buckets according to the partitioner.
    """

    def partitioner(row: List[str]):
        return int(hashlib.sha256(row[0].encode("utf-8") + len(row[0]).to_bytes(8, 'big') + key +
                                  len(key).to_bytes(4, 'big')).hexdigest(),
                   base=16)

    return partitioner
