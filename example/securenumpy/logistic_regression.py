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

import numpy as np
import petace.securenumpy as snp
import petace.secureml as sml
from petace.network import NetParams, NetScheme, NetFactory
from petace.backend.duet import DuetVM
from petace.engine import PETAceEngine


class LogisticRegression:

    def __init__(self, lr=0.01, max_iter=100, fit_intercept=True):
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z):
        return sml.sigmoid(z)

    def _add_intercept(self, X):
        intercept = snp.ones((X.shape[0], 1))
        return snp.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.theta = snp.zeros(X.shape[1])

        for i in range(self.max_iter):
            print(f"round: {i}")
            z = snp.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = snp.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._sigmoid(snp.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


if __name__ == '__main__':
    # load network setting
    import argparse

    parser = argparse.ArgumentParser(description='PETAce-Duet demo.')
    parser.add_argument("-p", "--party", type=int, help="which party")
    parser.add_argument("--port0", type=int, help="port of party 0, defalut 8089", default=8089)
    parser.add_argument("--port1", type=int, help="port of party 1, defalut 8090", default=8090)
    parser.add_argument("--host", type=str, help="host of this party", default="127.0.0.1")

    args = parser.parse_args()
    party = args.party
    port0 = args.port0
    port1 = args.port1
    host = args.host

    net_params = NetParams()
    if party == 0:
        net_params.remote_addr = host
        net_params.remote_port = port1
        net_params.local_port = port0
    else:
        net_params.remote_addr = host
        net_params.remote_port = port0
        net_params.local_port = port1

    # init mpc engine
    net = NetFactory.get_instance().build(NetScheme.SOCKET, net_params)
    duet = DuetVM(net, party)
    engine = PETAceEngine(duet)
    snp.set_engine(engine)

    # prepare data
    np.random.seed(43)
    if party == 0:
        x_plain = np.random.random((1000, 10))
        y_plain = None
    else:
        x_plain = None
        y_plain = np.random.randint(2, size=1000)

    x_cipher = snp.array(x_plain, party=0)
    y_cipher = snp.array(y_plain, party=1)

    clf = LogisticRegression(max_iter=5)
    clf.fit(x_cipher, y_cipher)

    y_pred = clf.predict(x_cipher)
    y_pred_plain = y_pred.reveal_to(0)
    if party == 0:
        print("The prediction:", y_pred_plain[:100])
