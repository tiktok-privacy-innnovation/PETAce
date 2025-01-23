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

from petace.securenumpy import SecureArray, get_engine


def groupby_sum(x: SecureArray, encoding: SecureArray) -> SecureArray:
    engine = get_engine()
    output = engine.new_share((x.shape[1], encoding.shape[1]), x.dtype)
    engine.execute_code("groupby_sum", [x.buffer, encoding.buffer, output])
    return SecureArray(output)


def groupby_count(x: SecureArray, encoding: SecureArray) -> SecureArray:
    engine = get_engine()
    output = engine.new_share((x.shape[1], encoding.shape[1]), x.dtype)
    engine.execute_code("groupby_count", [x.buffer, encoding.buffer, output])
    return SecureArray(output)


def groupby_max(x: SecureArray, encoding: SecureArray) -> SecureArray:
    engine = get_engine()
    output = engine.new_share((x.shape[1], encoding.shape[1]), x.dtype)
    engine.execute_code("groupby_max", [x.buffer, encoding.buffer, output])
    return SecureArray(output)


def groupby_min(x: SecureArray, encoding: SecureArray) -> SecureArray:
    engine = get_engine()
    output = engine.new_share((x.shape[1], encoding.shape[1]), x.dtype)
    engine.execute_code("groupby_min", [x.buffer, encoding.buffer, output])
    return SecureArray(output)
