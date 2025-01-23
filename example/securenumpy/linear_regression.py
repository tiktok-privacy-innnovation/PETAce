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

import numpy as numpy_np
import petace.securenumpy as np
from petace.network import NetParams, NetScheme, NetFactory
from petace.backend.duet import DuetVM
from petace.engine import PETAceEngine


class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        # Add bias term (intercept) to X
        X_b = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Number of training samples and features
        m, n = X_b.shape

        # Initialize weights (theta) to zeros
        self.theta = np.zeros(n)

        # Gradient Descent
        for _ in range(self.n_iterations):
            gradients = (1 / m) * np.dot(X_b.T, np.dot(X_b, self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        # Add bias term (intercept) to X
        X_b = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return np.dot(X_b, self.theta)

    def get_params(self):
        return self.theta


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
    np.set_engine(engine)

    # prepare data
    numpy_np.random.seed(43)
    if party == 0:
        x_plain = numpy_np.random.random((1000, 10))
        y_plain = None
    else:
        x_plain = None
        y_plain = numpy_np.random.random(1000)
    x_cipher = np.array(x_plain, party=0)
    y_cipher = np.array(y_plain, party=1)

    clf = LinearRegressionGD(n_iterations=5)
    clf.fit(x_cipher, y_cipher)

    y_pred = clf.predict(x_cipher)
    y_pred_plain = y_pred.reveal_to(0)
    if party == 0:
        print("The prediction:", y_pred_plain[:100])
