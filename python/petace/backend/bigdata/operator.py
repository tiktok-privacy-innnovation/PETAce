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


class Operator:

    map_func = {
        "share", "reveal", "add", "sub", "mul", "div", "and", "or", "xor", "not", "gt", "lt", "ge", "eq", "multiplexer"
    }
    reduce_func = {"max", "min", "sum"}
    aggregate_func = {"groupby_max": "max", "groupby_min": "min", "groupby_sum": "sum", "groupby_count": "sum"}
    onehot_func = {"onehot_encode"}
    unique_func = {"unique"}

    def is_map(self, func):
        return func in Operator.map_func

    def is_reduce(self, func):
        return func in Operator.reduce_func

    def is_aggregate(self, func):
        return func in Operator.aggregate_func

    def is_onehot(self, func):
        return func in Operator.onehot_func

    def is_unique(self, func):
        return func in Operator.unique_func
