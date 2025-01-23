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
    solo
    GIT_REPOSITORY https://github.com/tiktok-privacy-innovation/PETAce-Solo.git
    GIT_TAG        069ae987a1fafe77be3d5d7b5bcdd20697ced378 # 0.4.0
)
FetchContent_GetProperties(solo)

if(NOT solo_POPULATED)
    FetchContent_Populate(solo)

    set(SOLO_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SOLO_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SOLO_BUILD_BENCH OFF CACHE BOOL "" FORCE)
    set(SOLO_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SOLO_USE_IPCL OFF CACHE BOOL "" FORCE)

    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_SOLO)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_SOLO)

    add_subdirectory(
        ${solo_SOURCE_DIR}
        ${solo_SOURCE_DIR}/../solo-build
        EXCLUDE_FROM_ALL)
endif()
