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

cmake_minimum_required(VERSION 3.14)

#######################################################
# Project PETAce includes the following components:   #
#   1. PETAce Python wrapper library                  #
#######################################################

# [OPTION] CMAKE_BUILD_TYPE (DEFAULT: "Release")
# Select from Release, Debug, MiniSizeRel, or RelWithDebInfo.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

project(PETAce VERSION 0.4.0 LANGUAGES CXX C)

########################
# Global configuration #
########################

# CMake modules
include(CMakeDependentOption)
include(CMakePushCheckState)
include(CheckIncludeFiles)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

# Custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(PETAceCustomMacros)

# In Debug mode, define PETACE_DEBUG.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(PETACE_DEBUG ON)
else()
    set(PETACE_DEBUG OFF)
endif()
message(STATUS "PETACE debug mode: ${PETACE_DEBUG}")

# In Debug mode, enable extra compiler flags.
include(EnableDebugFlags)

# Always build position-independent-code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# [OPTION] PETACE_USE_CXX17 (default: OFF)
# Use C++17, use C++14 otherwise.
set(PETACE_USE_CXX17_OPTION_STR "Use C++17")
option(PETACE_USE_CXX17 ${PETACE_USE_CXX17_OPTION_STR} OFF)
message(STATUS "PETACE_USE_CXX17: ${PETACE_USE_CXX17}")
# Enable features from C++17 if available, disable features if set to OFF.
include(EnableCXX17)

# Add default files and directories.
include(GNUInstallDirs)

# Runtime path
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Build tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(PETACE_THIRDPARTY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

# Installation tree
set(PETACE_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/PETAce-${PETACE_VERSION_MAJOR}.${PETACE_VERSION_MINOR})
set(PETACE_INCLUDES_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/PETAce-${PETACE_VERSION_MAJOR}.${PETACE_VERSION_MINOR})
set(PETACE_THIRDPARTY_INCLUDES_INSTALL_DIR ${PETACE_INCLUDES_INSTALL_DIR}/thirdparty)

# Make the install target depend on the all target.
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY OFF)

# Supported target operating systems are Linux and macOS.
if (NOT DEFINED LINUX)
    if (UNIX AND NOT APPLE AND NOT CYGWIN AND NOT MINGW)
        set(LINUX ON)
    endif()
endif()
if (UNIX AND APPLE)
    set(MACOS ON)
endif()
if (NOT LINUX AND NOT MACOS)
    message(FATAL_ERROR "Supported target operating systems are Linux and macOS")
endif()

# Only support x86_64 and arm64
set(CMAKE_REQUIRED_QUIET_OLD ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_runs("
    #if defined(__aarch64__)
        int main() {
            return 0;
        }
    #else
        #error
    #endif
    "
    PETACE_ARM64
)
check_cxx_source_runs("
    #if defined(__amd64)
        int main() {
            return 0;
        }
    #else
        #error
    #endif
    "
    PETACE_AMD64
)
set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_OLD})
if (NOT PETACE_AMD64 AND NOT PETACE_ARM64)
    message(FATAL_ERROR "Supported target architectures are x86_64 and arm64")
endif()

add_compile_options(-msse4.2 -Wno-ignored-attributes -mavx)

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lrt")

#########################
# External dependencies #
#########################

# [OPTION] PETACE_BUILD_DEPS (DEFAULT: ON)
# Download and build dependencies if set to ON.
# Look for dependencies using find_package, otherwise.
set(PETACE_BUILD_DEPS_OPTION_STR "Automatically download and build unmet dependencies")
option(PETACE_BUILD_DEPS ${PETACE_BUILD_DEPS_OPTION_STR} ON)
message(STATUS "PETACE_BUILD_DEPS: ${PETACE_BUILD_DEPS}")

if(PETACE_BUILD_DEPS)
    include(FetchContent)
    mark_as_advanced(FETCHCONTENT_BASE_DIR)
    mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_QUIET)
endif()

# Require Threads::Threads
if(NOT TARGET Threads::Threads)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    if(NOT Threads_FOUND)
        message(FATAL_ERROR "Threads: not found, please download and install manually")
    else()
        message(STATUS "Threads: found")
    endif()
endif()

# PETAce-Network::network
find_package(PETAce-Network 0.3 QUIET CONFIG)
if(PETAce-Network_FOUND)
    message(STATUS "PETAce-Network: found")
    if(PETAce-Network_STATIC_FOUND)
        set(network "PETAce-Network::network")
    else()
        set(network "PETAce-Network::network_shared")
    endif()
else()
    if(PETACE_BUILD_DEPS)
        message(STATUS "PETAce-Network: download ...")
        petace_fetch_thirdparty_content(PETAce-Network)
        if(TARGET network)
            set(network "network")
        else()
            set(network "network_shared")
        endif()
        set(PETACE_BUILD_NETWORK TRUE CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "PETAce-Network: not found, please download and install manually")
    endif()
endif()

# PETAce-Solo::solo
find_package(PETAce-Solo 0.3 QUIET CONFIG)
if(PETAce-Solo_FOUND)
    message(STATUS "PETAce-Solo: found")
    if(PETAce-Solo_STATIC_FOUND)
        set(solo "PETAce-Solo::solo")
    else()
        set(solo "PETAce-Solo::solo_shared")
    endif()
else()
    if(PETACE_BUILD_DEPS)
        message(STATUS "PETAce-Solo: download ...")
        petace_fetch_thirdparty_content(PETAce-Solo)
        if(TARGET solo)
            set(solo "solo")
        else()
            set(solo "solo_shared")
        endif()
        set(PETACE_BUILD_SOLO TRUE CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "PETAce-Solo: not found, please download and install manually")
    endif()
endif()

# PETAce-Verse::verse
find_package(PETAce-Verse 0.3 QUIET CONFIG)
if(PETAce-Verse_FOUND)
    message(STATUS "PETAce-Verse: found")
    if(PETAce-Verse_STATIC_FOUND)
        set(verse "PETAce-Verse::verse")
    else()
        set(verse "PETAce-Verse::verse_shared")
    endif()
else()
    if(PETACE_BUILD_DEPS)
        message(STATUS "PETAce-Verse: download ...")
        petace_fetch_thirdparty_content(PETAce-Verse)
        if(TARGET verse)
            set(verse "verse")
        else()
            set(verse "verse_shared")
        endif()
        set(PETACE_BUILD_VERSE TRUE CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "PETAce-Verse: not found, please download and install manually")
    endif()
endif()

# PETAce-Duet::duet
find_package(PETAce-Duet 0.3 QUIET CONFIG)
if(PETAce-Duet_FOUND)
    message(STATUS "PETAce-Duet: found")
    if(PETAce-Duet_STATIC_FOUND)
        set(duet "PETAce-Duet::duet")
    else()
        set(duet "PETAce-Duet::duet_shared")
    endif()
else()
    if(PETACE_BUILD_DEPS)
        message(STATUS "PETAce-Duet: download ...")
        petace_fetch_thirdparty_content(PETAce-Duet)
        if(TARGET duet)
            set(duet "duet")
        else()
            set(duet "duet_shared")
        endif()
        set(PETACE_BUILD_DUET TRUE CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "PETAce-Duet: not found, please download and install manually")
    endif()
endif()

# PETAce-SetOps::setops
find_package(PETAce-SetOps 0.3 QUIET CONFIG)
if(PETAce-SetOps_FOUND)
    message(STATUS "PETAce-SetOps: found")
    if(PETAce-SetOps_STATIC_FOUND)
        set(setops "PETAce-SetOps::setops")
    else()
        set(setops "PETAce-SetOps::setops_shared")
    endif()
else()
    if(PETACE_BUILD_DEPS)
        message(STATUS "PETAce-SetOps: download ...")
        petace_fetch_thirdparty_content(PETAce-SetOps)
        if(TARGET setops)
            set(setops "setops")
        else()
            set(setops "setops_shared")
        endif()
        set(PETACE_BUILD_SETOPS TRUE CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "PETAce-SetOps: not found, please download and install manually")
    endif()
endif()

##################
# PETACE python #
##################

# [OPTION] PETACE_BUILD_PYTHON (DEFAULT: ON)
# Build a shared library if set to ON.
set(PETACE_BUILD_PYTHON_STR "Build python shared library")
option(PETACE_BUILD_PYTHON ${PETACE_BUILD_PYTHON_STR} ON)
message(STATUS "PETACE_BUILD_PYTHON: ${PETACE_BUILD_PYTHON}")

if(PETACE_BUILD_PYTHON)
    add_subdirectory(python)
endif()
