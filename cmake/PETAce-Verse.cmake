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
    verse
    GIT_REPOSITORY https://github.com/tiktok-privacy-innovation/PETAce-Verse.git
    GIT_TAG        1514f98f6bee3cfaaadd919e2f4f1e45c80f6e62 # 0.4.0
)
FetchContent_GetProperties(verse)

if(NOT verse_POPULATED)
    FetchContent_Populate(verse)

    set(VERSE_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(VERSE_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(VERSE_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(VERSE_BUILD_BENCH OFF CACHE BOOL "" FORCE)

    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_VERSE)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_VERSE)

    add_subdirectory(
        ${verse_SOURCE_DIR}
        ${verse_SOURCE_DIR}/../verse-build
        EXCLUDE_FROM_ALL)
endif()
