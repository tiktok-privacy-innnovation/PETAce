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

from typing import Union
import json
import numpy as np
import pandas as pd
from petace.duet.pyduet import DuetVM, Instruction
from petace.backend.bigdata._type import BigDataPrivateType, BigDataShareType, BigDataPublicType
from petace.backend.bigdata.crypto_data import Private, Shared, Public
from petace.backend.bigdata.operator import Operator
from petace.backend.bigdata.network_manger import NetworkManger, NetworkType
from petace.backend import VMBase, PETAceBackendType
from crypto_data_engine import PREDEFINED, CSVDataset, Context


class BigDataVM(VMBase):
    """
    PETAce's BigData VM. It has compatible interfaces with Petace Duet VM.

    Internally, a parallelized engine (multi-process, Spark) is used to call instances of PETAce
    on each partition of the data to support big data computations.

    Parameters:
    ----------
    party : int
        The party id.
    engine_ctx_params : dict
        The params of the parallelized engine context.
    net_scheme : str
        The network scheme, agent or socket
    net_params : dict
        The params of network
    partition_num : int
        Partition number of data
    """

    def __init__(
        self,
        party,
        engine_ctx_params,
        net_scheme,
        net_params,
        partition_num=3,
    ):
        self.party = party
        self.engine_ctx = Context(**engine_ctx_params)
        self.count = 0
        self.operator = Operator()
        self.partition_num = partition_num
        self.common_network = NetworkManger(NetworkType.SOCKET if net_scheme == 'socket' else NetworkType.AGENT,
                                            net_params[str(party)]).get_network(0, -1)
        self.kwargs = {
            'scheme': net_scheme,
            'party': str(party),
            'parties': net_params,
        }

    @property
    def type(self):
        return PETAceBackendType.BigData

    def __del__(self):
        self.engine_ctx.close()

    @staticmethod
    def get_engine(party, count, index, network_manger):
        net = network_manger.get_network(count, index)
        vm = DuetVM(net, party)
        return vm

    @staticmethod
    def set_input_data(vm, party, input_party, data_type, input, snp_data):
        if input_party in {0, 1}:
            if data_type == BigDataPrivateType.DOUBLE:
                addr = vm.new_private_double_matrix(input_party)
            elif data_type == BigDataPrivateType.BOOL:
                addr = vm.new_private_bool_matrix(input_party)
            else:
                raise RuntimeError("set private data type error")
            if input_party == party:
                if data_type == BigDataPrivateType.DOUBLE:
                    array = np.array(input.read(), dtype=np.float64)
                    vm.set_private_double_matrix(array, addr)
                elif data_type == BigDataPrivateType.BOOL:
                    array = np.array(input.read()) == "True"
                    vm.set_private_bool_matrix(array, addr)
            snp_data.append(addr)
        if input_party == -1:
            array = np.array(input.read(), dtype=np.int64)
            if data_type == BigDataShareType.DOUBLE:
                addr = vm.new_airth_matrix()
                vm.set_airth_share_matrix(array, addr)
            elif data_type == BigDataShareType.BOOL:
                addr = vm.new_bool_matrix()
                vm.set_boolean_share_matrix(array, addr)
            else:
                raise RuntimeError("set share data type error")
            snp_data.append(addr)
        if input_party == 2:
            if data_type == BigDataPublicType.DOUBLE:
                addr = vm.new_public_double_matrix()
                array = np.array(input.read(), dtype=np.float64)
                vm.set_public_double_matrix(array, addr)
            elif data_type == BigDataPublicType.BOOL:
                addr = vm.new_public_bool_matrix()
                array = np.array(input.read(), dtype=np.bool_)
                vm.set_public_bool_matrix(array, addr)
            elif data_type == BigDataPublicType.IINT:
                addr = vm.new_public_index()
                data = int(input.read())
                vm.set_public_index(data, addr)
            elif data_type == BigDataPublicType.IDOUBLE:
                addr = vm.new_public_double()
                data = float(input.read())
                vm.set_public_double(data, addr)
            else:
                raise RuntimeError("set public data type error")
            snp_data.append(addr)

    @staticmethod
    def set_output_data(vm, input_party, data_type, snp_data):
        if input_party in {0, 1}:
            if data_type == BigDataPrivateType.DOUBLE:
                addr = vm.new_private_double_matrix(input_party)
            elif data_type == BigDataPrivateType.BOOL:
                addr = vm.new_private_bool_matrix(input_party)
            else:
                raise RuntimeError("private input data type not supported")
            snp_data.append(addr)
        if input_party == -1:
            if data_type == BigDataShareType.DOUBLE:
                addr = vm.new_airth_matrix()
            elif data_type == BigDataShareType.BOOL:
                addr = vm.new_bool_matrix()
            else:
                raise RuntimeError("share input data type not supported")
            snp_data.append(addr)
        if input_party == 2:
            raise RuntimeError("public input data type not supported")

    @staticmethod
    def exec_code_by_partition(vm, inputs, outputs, code, data_party, data_type, **kwargs):
        # pylint: disable=too-many-try-statements
        try:
            input_num = len(inputs)
            snp_data = []
            party = int(kwargs['party'])

            for i, item in enumerate(data_party):
                if i < input_num:
                    BigDataVM.set_input_data(vm, party, item, data_type[i], inputs[i], snp_data)
                else:
                    BigDataVM.set_output_data(vm, item, data_type[i], snp_data)
            inst = Instruction([code] + data_type)
            vm.exec_code(inst, snp_data)

            output_num = len(outputs)
            output_data_type = data_type[input_num:]
            output_party = data_party[input_num:]
            output_addr = snp_data[input_num:]

            for i in range(output_num):
                if output_party[i] in {0, 1}:
                    if output_party[i] == party:
                        if output_data_type[i] == BigDataPrivateType.DOUBLE:
                            ret = vm.get_private_double_matrix(output_addr[i])
                        elif output_data_type[i] == BigDataPrivateType.BOOL:
                            ret = vm.get_private_bool_matrix(output_addr[i])
                        else:
                            raise RuntimeError("private output get data type not supported")
                        if code.startswith("groupby"):
                            ret = ret.reshape((-1, 1))
                        outputs[i].save(ret.tolist())
                    else:
                        pass
                if output_party[i] == -1:
                    if output_data_type[i] == BigDataShareType.DOUBLE:
                        ret = vm.get_airth_share_matrix(output_addr[i])
                    elif output_data_type[i] == BigDataShareType.BOOL:
                        ret = vm.get_boolean_share_matrix(output_addr[i])
                    else:
                        raise RuntimeError("share output get data type not supported")
                    if code.startswith("groupby"):
                        ret = ret.reshape((-1, 1))
                    outputs[i].save(ret.tolist())
                if output_party[i] == 2:
                    raise RuntimeError("public output get data type not supported")

        except Exception as e:
            print(e)
            raise e
        return True

    @staticmethod
    def get_one_hot_encode(_, inputs, outputs, group_col_dtype, *__, **___):
        if group_col_dtype != BigDataPrivateType.DOUBLE:
            raise RuntimeError("now only support onehot row is float type")
        unique_data = json.loads(inputs[0].read())
        unique_data = sorted([float(item[0]) for item in unique_data])
        data_to_onehot = pd.DataFrame(inputs[1].read(), columns=["value"]).astype(float)
        one_hot_encoded = pd.get_dummies(data_to_onehot["value"])
        one_hot_encoded = one_hot_encoded.reindex(columns=unique_data, fill_value=0).astype(int)

        outputs[0].save(one_hot_encoded.values.tolist())
        return True

    @staticmethod
    def get_unique(_, inputs, outputs, group_col_dtype, *__, **___):
        if group_col_dtype != BigDataPrivateType.DOUBLE:
            raise RuntimeError("now only support onehot row is float type")
        unique_data = json.loads(inputs[0].read())
        unique_data = sorted([float(item[0]) for item in unique_data])
        outputs[0].save([[item] for item in unique_data])
        return True

    @staticmethod
    def merge_group_by_ret(vm, inputs, outputs, code, partition_num, **_):
        data = inputs[0].read()
        if len(data) % partition_num != 0:
            raise RuntimeError("data length must be divisible by partition_num")
        one_array_len = int(len(data) / partition_num)

        data_to_merge = []
        for i in range(partition_num):
            tmp_array = np.array(data[i * one_array_len:(i + 1) * one_array_len]).astype(np.int64)
            data_to_merge.append(vm.new_airth_matrix())
            vm.set_airth_share_matrix(tmp_array, data_to_merge[-1])

        if code == "sum":
            ret = vm.new_airth_matrix()
            vm.exec_code(Instruction(["add", "am", "am", "am"]), [data_to_merge[0], data_to_merge[1], ret])
            for i in range(2, partition_num):
                vm.exec_code(Instruction(["add", "am", "am", "am"]), [ret, data_to_merge[i], ret])
            ret = vm.get_airth_share_matrix(ret)
            outputs[0].save(ret.tolist())
        max_min_map = {"max": "ge", "min": "lt"}
        if code in max_min_map:
            ret = vm.new_airth_matrix()
            tmp = vm.new_bool_matrix()
            vm.exec_code(Instruction([max_min_map[code], "am", "am", "bm"]), [data_to_merge[0], data_to_merge[1], tmp])
            vm.exec_code(Instruction(["multiplexer", "bm", "am", "am", "am"]),
                         [tmp, data_to_merge[1], data_to_merge[0], ret])
            for i in range(2, partition_num):
                tmp = vm.new_bool_matrix()
                vm.exec_code(Instruction([max_min_map[code], "am", "am", "bm"]), [ret, data_to_merge[i], tmp])
                vm.exec_code(Instruction(["multiplexer", "bm", "am", "am", "am"]), [tmp, data_to_merge[i], ret, ret])
            ret = vm.get_airth_share_matrix(ret)
            outputs[0].save(ret.tolist())

        return True

    def exec_code(self, code_all, data):
        # pylint: disable=too-many-statements
        code = code_all[0]
        kwargs = self.kwargs
        kwargs['count'] = self.count
        if self.operator.is_map(code):
            tmp_map_inner_data = [item.data for item in data]
            self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_map_inner_data[0:-1],
                                    tmp_map_inner_data[-1:], code, [item.party for item in data],
                                    [item.dtype for item in data], **kwargs)
            self.count += 1
        elif self.operator.is_reduce(code):
            if data[0].data.partition_num == 1:
                tmp_reduce_inner_data = [item.data for item in data]
                self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_reduce_inner_data[0:-1],
                                        tmp_reduce_inner_data[-1:], code, [item.party for item in data],
                                        [item.dtype for item in data], **kwargs)
                self.count += 1
            else:
                tmp_data = data
                tmp_reduce_data = [self.new_airth_matrix(), data[-1]]
                tmp_data[-1] = tmp_reduce_data[0]
                tmp_reduce_inner_data_0 = [item.data for item in tmp_data]
                self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_reduce_inner_data_0[0:-1],
                                        tmp_reduce_inner_data_0[-1:], code, [item.party for item in tmp_data],
                                        [item.dtype for item in tmp_data], **kwargs)
                self.count += 1
                self.engine_ctx.compute(PREDEFINED.REPARTITION, None, [tmp_reduce_data[0].data], [], 1)
                tmp_reduce_inner_data_1 = [item.data for item in tmp_reduce_data]
                self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_reduce_inner_data_1[0:-1],
                                        tmp_reduce_inner_data_1[-1:], code, [item.party for item in tmp_reduce_data],
                                        [item.dtype for item in tmp_reduce_data], **kwargs)
                self.count += 1
        elif self.operator.is_onehot(code):
            group_by_col = data[0]
            onehot_ret = data[1]
            if self.party == group_by_col.party:
                unique_data = self.engine_ctx.scalar_dataset(value=0.0, partition_num=group_by_col.data.partition_num)
                self.engine_ctx.compute(PREDEFINED.GET_UNIQUE_ROWS_FROM_CSV, None, [group_by_col.data], [unique_data],
                                        0)
                self.engine_ctx.compute(None,
                                        self.get_one_hot_encode, [unique_data, group_by_col.data], [onehot_ret.data],
                                        group_by_col.dtype,
                                        vm_mode='none')
        elif self.operator.is_unique(code):
            group_by_col = data[0]
            unique_ret = data[1]
            self.engine_ctx.compute(PREDEFINED.REPARTITION, None, [unique_ret.data], [], 1)
            if self.party == group_by_col.party:
                unique_data = self.engine_ctx.scalar_dataset(value=0.0, partition_num=1)
                self.engine_ctx.compute(PREDEFINED.GET_UNIQUE_ROWS_FROM_CSV, None, [group_by_col.data], [unique_data],
                                        0)
                self.engine_ctx.compute(None,
                                        self.get_unique, [unique_data], [unique_ret.data],
                                        group_by_col.dtype,
                                        vm_mode='none')
        elif self.operator.is_aggregate(code):
            if data[0].data.partition_num == 1:
                tmp_agg_data = [item.data for item in data]
                self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_agg_data[0:-1], tmp_agg_data[-1:], code,
                                        [item.party for item in data], [item.dtype for item in data], **kwargs)
                self.count += 1
            else:
                tmp_data = data
                tmp_agg_data = [self.new_airth_matrix(), data[-1]]
                tmp_data[-1] = tmp_agg_data[0]
                partition_num = tmp_agg_data[0].data._partition_num
                tmp_agg_inner_data_0 = [item.data for item in tmp_data]
                self.engine_ctx.compute(None, self.exec_code_by_partition, tmp_agg_inner_data_0[0:-1],
                                        tmp_agg_inner_data_0[-1:], code, [item.party for item in tmp_data],
                                        [item.dtype for item in tmp_data], **kwargs)
                self.count += 1
                self.engine_ctx.compute(PREDEFINED.REPARTITION, None, [tmp_agg_data[0].data], [], 1)
                if tmp_agg_data[-1].data.partition_num != 1:
                    self.engine_ctx.compute(PREDEFINED.REPARTITION, None, [tmp_agg_data[-1].data], [], 1)
                tmp_agg_inner_data_1 = [item.data for item in tmp_agg_data]
                self.engine_ctx.compute(None, self.merge_group_by_ret, tmp_agg_inner_data_1[0:-1],
                                        tmp_agg_inner_data_1[-1:], Operator.aggregate_func[code], partition_num,
                                        **kwargs)
                self.count += 1

        else:
            raise RuntimeError(f"unsupported operator: {code}")

    def run(self):
        try:
            self.engine_ctx.execute()
        except Exception as e:
            print(e)
            raise e

    def close(self):
        self.engine_ctx.close()

    def new_csv_dataset(self, path=None, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return self.engine_ctx.csv_dataset(file_path=path, partition_num=p)

    def refine_engine_ctx(self):
        self.engine_ctx.clear()
        self.engine_ctx.close()
        self.engine_ctx = Context(work_directory=self.engine_ctx._work_dir)

    def new_airth_matrix(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Shared(self.engine_ctx.csv_dataset(partition_num=p), BigDataShareType.DOUBLE)

    def new_bool_matrix(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Shared(self.engine_ctx.csv_dataset(partition_num=p), BigDataShareType.BOOL)

    def new_public_double_matrix(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Public(self.engine_ctx.csv_dataset(partition_num=p), BigDataPublicType.DOUBLE)

    def new_public_double(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Public(self.engine_ctx.scalar_dataset(value=0, partition_num=p), BigDataPublicType.IDOUBLE)

    def new_public_index(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Public(self.engine_ctx.scalar_dataset(value=0, partition_num=p), BigDataPublicType.IINT)

    def new_public_bool_matrix(self, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        return Public(self.engine_ctx.csv_dataset(partition_num=p), BigDataPublicType.BOOL)

    def new_private_double_matrix(self, party, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        if party == self.party:
            return Private(party, self.engine_ctx.csv_dataset(partition_num=p), BigDataPrivateType.DOUBLE)
        return Private(party, self.engine_ctx.csv_dataset(placeholder=True, partition_num=p), BigDataPrivateType.DOUBLE)

    def new_private_bool_matrix(self, party, partition_num=0):
        p = self.partition_num if partition_num == 0 else partition_num
        if party == self.party:
            return Private(party, self.engine_ctx.csv_dataset(partition_num=p), BigDataPrivateType.BOOL)
        return Private(party, self.engine_ctx.csv_dataset(placeholder=True, partition_num=p), BigDataPrivateType.BOOL)

    def _set_data(self, data, crypto_data):
        if isinstance(data, np.ndarray):
            crypto_data.set_array(data, index=False, header=False)
        elif isinstance(data, str):
            crypto_data.data.file_path = data
        elif isinstance(data, CSVDataset):
            crypto_data.data = data
        else:
            raise RuntimeError(f"unsupported data type: {type(data)}")

    def set_private_double_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Private"):
        if crypto_data.party == self.party:
            self._set_data(data, crypto_data)

    def set_private_bool_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Private"):
        if crypto_data.party == self.party:
            self._set_data(data, crypto_data)

    def set_public_double_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Public"):
        self._set_data(data, crypto_data)

    def set_public_bool_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Public"):
        self._set_data(data, crypto_data)

    def set_public_double(self, value: float, data: "Public"):
        data.set_scalar(value)

    def set_public_index(self, value: int, data: "Public"):
        data.set_scalar(value)

    def set_airth_share_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Shared"):
        self._set_data(data, crypto_data)

    def set_boolean_share_matrix(self, data: Union[np.ndarray, str, "CSVDataset"], crypto_data: "Shared"):
        self._set_data(data, crypto_data)

    def get_private_double_matrix(self, data: "Private"):
        if self.party == data.party:
            return data.get_array(header=None, index_col=False).astype(np.float64)
        return np.empty((0, 0), dtype=np.float64)

    def get_private_bool_matrix(self, data: "Private"):
        if self.party == data.party:
            return data.get_array(header=None, index_col=False).astype(np.bool_)
        return np.empty((0, 0), dtype=np.bool_)

    def get_airth_share_matrix(self, data: "Shared"):
        return data.get_array(header=None, index_col=False).astype(np.int64)

    def get_boolean_share_matrix(self, data: "Shared"):
        return data.get_array(header=None, index_col=False).astype(np.int64)

    def delete_data(self, data):
        # no need
        pass

    def is_registr_empty(self):
        # no need
        return True

    def party_id(self):
        return self.party

    def get_private_double_matrix_shape(self, data: "Private"):
        return data.data.shape()

    def get_public_double_matrix_shape(self, data: "Public"):
        return data.data.shape()

    def get_airth_share_matrix_shape(self, data: "Shared"):
        return data.data.shape()

    def get_bool_share_matrix_shape(self, data: "Shared"):
        return data.data.shape()

    def public_double_matrix_block(self, data_0, data_1, begin_row, begin_col, row_num, col_num):
        raise RuntimeError("not support")

    def private_double_matrix_block(self, data_0, data_1, begin_row, begin_col, row_num, col_num):
        raise RuntimeError("not support")

    def airth_share_matrix_block(self, data_0, data_1, begin_row, begin_col, row_num, col_num):
        raise RuntimeError("not support")

    def bool_share_matrix_block(self, data_0, data_1, begin_row, begin_col, row_num, col_num):
        raise RuntimeError("not support")

    def private_double_vstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def public_double_vstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def airth_share_vstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def bool_share_vstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def private_double_hstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def public_double_hstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def airth_share_hstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def bool_share_hstack(self, in_data_0, in_data_1, out_data):
        raise RuntimeError("not support")

    def send_buffer(self, data):
        self.common_network.send_data(data, len(data))

    def recv_buffer(self, data_size):
        buffer = bytearray(data_size)
        self.common_network.recv_data(buffer, data_size)
        return buffer
