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

from abc import ABC
import os
import traceback
from multiprocessing import Process

from petace.network import NetParams, NetScheme, NetFactory
import petace.securenumpy as snp
from petace.backend import PETAceBackendType
from petace.backend.duet import DuetVM
from petace.engine import PETAceEngine
from petace.backend.bigdata import BigDataVM


class PETAceTestException(Exception):

    def __init__(self, party, message) -> None:
        self.party = party
        self.message = message

    def __str__(self):
        return f"party {self.party} test failed: {self.message}"


def run_test(test_func):
    processes = []

    for i in range(2):
        p = Process(target=test_func, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for p in processes:
        if p.exitcode != 0:
            # pylint: disable=broad-exception-raised
            raise Exception(f"party {p.exitcode} test failed")


def init_network(party):
    net_params = NetParams()
    ip1 = "127.0.0.1"
    port1 = 8890
    ip2 = "127.0.0.1"
    port2 = 8891
    if party == 0:
        net_params.remote_addr = ip1
        net_params.remote_port = port1
        net_params.local_port = port2
    else:
        net_params.remote_addr = ip2
        net_params.remote_port = port2
        net_params.local_port = port1
    net = NetFactory.get_instance().build(NetScheme.SOCKET, net_params)
    return net


class SinglePartyTestBase(ABC):

    def run_test_all(self):
        for method in dir(self):
            if method.startswith("test_"):
                getattr(self, method)()


class TwoPartyTestBase(ABC):

    def run_test_all(self):
        run_test(self.run_process)

    def run_process(self, party):
        for method in dir(self):
            if method.startswith("test_"):
                getattr(self, method)(party)


class NetworkTestBase(ABC):

    def run_test_all(self):
        run_test(self.run_process)

    def run_process(self, party):
        self.net = init_network(party)
        for method in dir(self):
            if method.startswith("test_"):
                getattr(self, method)(party)


def init_bigdata_engine(party):
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

    bigdata = BigDataVM(party, {"work_directory": dir + "/bigdata/test_data"}, 'socket', parties, 1)
    engine = PETAceEngine(bigdata)
    return engine


def init_duet_engine(party):
    net = init_network(party)
    duet = DuetVM(net, party)
    engine = PETAceEngine(duet)
    return engine


class SnpTestBase(ABC):

    def run_test_all(self):
        run_test(self.run_process)

    def run_process(self, party):
        engineBackend = os.environ.get("PETACE_ENGINE_BACKEND", PETAceBackendType.Duet)
        print(f"engineBackend: {engineBackend}")
        if engineBackend == PETAceBackendType.BigData:
            engine = init_bigdata_engine(party)
        elif engineBackend == PETAceBackendType.Duet:
            engine = init_duet_engine(party)
        else:
            raise ValueError(f'engineBackend {engineBackend} is not supported.')
        snp.set_engine(engine)
        for method in dir(self):
            if method.startswith("test_"):
                try:
                    getattr(self, method)(engine.party_id())
                # pylint: disable=raise-missing-from
                except Exception as e:
                    traceback.print_stack()
                    raise PETAceTestException(party, e)


class SnpBigdataTestBase(ABC):

    def run_test_all(self):
        run_test(self.run_process)

    def run_process(self, party):
        engine = init_bigdata_engine(party)
        self.engine = engine
        snp.set_engine(engine)
        for method in dir(self):
            if method.startswith("test_"):
                getattr(self, method)(engine.party_id())

    def init_vm(self, party):
        self.engine = init_bigdata_engine(party)
        return self.engine

    def run_vm(self):
        self.engine.vm.run()

    def close_vm(self):
        self.engine.vm.close()
        snp.set_engine(None)
