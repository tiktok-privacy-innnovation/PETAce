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

FetchContent_Declare(
    setops
    GIT_REPOSITORY https://github.com/tiktok-privacy-innovation/PETAce-SetOps.git
    GIT_TAG        94c2f6ce503ebff128b823c533a6272c864e0f63 # 0.4.0
)
FetchContent_GetProperties(setops)

if(NOT setops_POPULATED)
    FetchContent_Populate(setops)

    set(SETOPS_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SETOPS_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SETOPS_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_SETOPS)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_SETOPS)

    add_subdirectory(
        ${setops_SOURCE_DIR}
        ${setops_SOURCE_DIR}/../setops-build
        EXCLUDE_FROM_ALL)
endif()
