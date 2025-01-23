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
    duet
    GIT_REPOSITORY https://github.com/tiktok-privacy-innovation/PETAce-Duet.git
    GIT_TAG        018c1b876e61c9b28a107b35f3ee0271cab43486 # 0.4.0
)
FetchContent_GetProperties(duet)

if(NOT duet_POPULATED)
    FetchContent_Populate(duet)

    set(DUET_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(DUET_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(DUET_BUILD_BENCH OFF CACHE BOOL "" FORCE)
    set(DUET_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_DUET)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_DUET)

    add_subdirectory(
        ${duet_SOURCE_DIR}
        ${duet_SOURCE_DIR}/../duet-build
        EXCLUDE_FROM_ALL)
endif()
