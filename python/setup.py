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

import os
import re
import sys
import setuptools


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join('.', *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        print("Unable to find version string.")
        sys.exit(-1)


def read_requirements():
    requirements = []
    _dependency_links = []
    root_path = os.path.abspath(os.path.dirname(__file__))
    with open(f'{root_path}/requirements.txt') as file:
        requirements = file.read().splitlines()
    for r in requirements:
        if r.startswith("--extra-index-url"):
            requirements.remove(r)
            _dependency_links.append(r)
    print("Requirements: ", requirements)
    print("Dependency: ", _dependency_links)
    return requirements, _dependency_links


install_requires, dependency_links = read_requirements()

setuptools.setup(
    name="petace",
    version=find_version("petace", "version.py"),
    author="PETAce",
    author_email="petace@tiktok.com",
    description="petace",
    url="",
    package_dir={"petace": "./petace"},
    package_data={
        "petace.network": ["*.so"],
        "petace.setops": ["*.so"],
        "petace.duet": ["*.so"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires="==3.*",
    install_requires=install_requires,
    dependency_links=dependency_links,
)
