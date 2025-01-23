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

from petace.engine import PETAceEngine


class GlobalPETAceEngine:
    engine = None


GLOBALEngine = GlobalPETAceEngine()


def set_engine(engine: PETAceEngine):
    GLOBALEngine.engine = engine


def get_engine() -> PETAceEngine:
    if GLOBALEngine.engine is None:
        raise RuntimeError("Global PETAceEngine is not initialized")
    return GLOBALEngine.engine
