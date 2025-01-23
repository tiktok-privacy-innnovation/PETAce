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
import petace.securenumpy as snp
from petace.engine import PETAceEngine
from petace.backend.bigdata import BigDataVM

if __name__ == '__main__':
    # load network setting
    import argparse

    parser = argparse.ArgumentParser(description='PETAce-Duet demo.')
    parser.add_argument("-p", "--party", type=int, help="which party")
    parser.add_argument("--port0", type=int, help="port of party 0, defalut 18000", default=18000)
    parser.add_argument("--port1", type=int, help="port of party 1, defalut 19000", default=19000)
    parser.add_argument("--host", type=str, help="host of this party", default="127.0.0.1")
    parser.add_argument('-et', "--engine_type", type=str, help="engine type", default="local")
    parser.add_argument('--core', type=int, help="test_num", default=2)
    parser.add_argument('--part', type=int, help="test_num", default=2)

    args = parser.parse_args()
    party = args.party
    port0 = args.port0
    port1 = args.port1
    host = args.host
    engine_type = args.engine_type
    core = args.core
    part = args.part

    dir = os.path.dirname(os.path.abspath(__file__))

    parties = {
        '0': {
            "remote_addr": host,
            "remote_port_base": port1,
            "local_port_base": port0
        },
        '1': {
            "remote_addr": host,
            "remote_port_base": port0,
            "local_port_base": port1
        }
    }
    engine_ctx_params = {"work_directory": dir + "/bigdata/test_data", 'engine_type': engine_type, 'max_workers': core}

    bigdata = BigDataVM(party, engine_ctx_params, "socket", parties, part)
    engine = PETAceEngine(bigdata)
    snp.set_engine(engine)
    if party == 0:
        # notice: The rows of data must bigger than both core and part
        # You could also put a path of a CSV file without header here.
        array0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64).reshape((3, 4))
        array1 = None
    else:
        array0 = None
        array1 = np.array([7, 2, 3, 4, 5, 6, 7, 10, 9, 10, 11, 12], dtype=np.float64).reshape((3, 4))
    c0 = snp.array(array0, 0)
    c1 = snp.array(array1, 1)
    c2 = c0 + c1
    ret = c2.reveal_to(0)
    # When input a CSV file as input, you can use
    ################################
    # ret = c2.reveal_to(0, False) #
    # print(ret.file_path)         #
    ################################
    # to get the result CSV file path
    if party == 0:
        print("ret:", ret)
