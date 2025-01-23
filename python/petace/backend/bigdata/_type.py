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


class PETAceBigDataType(ABC):

    @classmethod
    def support_types(cls):
        roles = set()
        for role_key, role in cls.__dict__.items():
            if role_key.startswith("__") and isinstance(role_key, str):
                continue

            roles.add(role)

        return roles


class BigDataPrivateType(PETAceBigDataType):
    DOUBLE = "pdm"
    BOOL = "pbm"


class BigDataShareType(PETAceBigDataType):
    DOUBLE = "am"
    BOOL = "bm"


class BigDataPublicType(PETAceBigDataType):
    DOUBLE = "cdm"
    BOOL = "cbm"
    IDOUBLE = "cd"  # single double value
    IINT = "ci"  # single int value
