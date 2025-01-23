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

from petace.securenumpy import get_engine
from crypto_data_engine import CSVDataset


def onehot_encode(input: "CSVDataset", party: int):
    vm = get_engine().vm
    pinput = vm.new_private_double_matrix(party)
    poutput = vm.new_private_double_matrix(party)
    if party == vm.party:
        vm.set_private_double_matrix(input, pinput)
    data_to_onehot_encode = [pinput, poutput]
    vm.exec_code(["onehot_encode", *[obj.dtype for obj in data_to_onehot_encode]], [pinput, poutput])
    return poutput.data


def unique(input: "CSVDataset", party: int):
    vm = get_engine().vm
    pinput = vm.new_private_double_matrix(party)
    poutput = vm.new_private_double_matrix(party, 1)
    if party == vm.party:
        vm.set_private_double_matrix(input, pinput)
    data_to_get_unique = [pinput, poutput]
    vm.exec_code(["unique", *[obj.dtype for obj in data_to_get_unique]], [pinput, poutput])
    return poutput.data
