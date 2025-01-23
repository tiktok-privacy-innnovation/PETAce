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
from crypto_data_engine import Context
from crypto_data_engine import DataType

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='PETAce-Duet demo.')
    parser.add_argument("-p", "--party", type=int, help="which party")
    parser.add_argument("--port0", type=int, help="port of party 0, default 15000", default=15000)
    parser.add_argument("--port1", type=int, help="port of party 1, default 16000", default=16000)
    parser.add_argument("--host", type=str, help="host of this party", default="127.0.0.1")
    parser.add_argument('-et', "--engine_type", type=str, help="engine type", default="local")

    args = parser.parse_args()
    party = args.party
    port0 = args.port0
    port1 = args.port1
    host = args.host
    engine_type = args.engine_type

    dir = os.path.dirname(os.path.abspath(__file__))

    engine_ctx_params = {"work_directory": dir + "/test_data", 'engine_type': engine_type}

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

    ctx = Context(**engine_ctx_params)
    setops = BigdataSetOps(party, ctx, "socket", parties)
    if party == 0:
        input_path = dir + "/data/dataset1.csv"
        input_ = ctx.csv_dataset(file_path=input_path, data_type=[DataType.String])
        output_path = dir + "/data/tmp/dataset1_output.csv"
        output = ctx.csv_dataset(file_path=output_path)
    else:
        input_path = dir + "/data/dataset2.csv"
        input_ = ctx.csv_dataset(file_path=input_path, data_type=[DataType.String])
        output_path = dir + "/data/tmp/dataset2_output.csv"
        output = ctx.csv_dataset(file_path=output_path)
    setops.run([input_, output], True, setops_scheme=PSIScheme.ECDH_PSI, sort_result=True)
    print("The size of the intersection is: ", output.shape()[0])
