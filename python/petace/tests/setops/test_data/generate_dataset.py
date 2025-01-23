# Copyright 2024 TikTok Pte. Ltd.
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

import random
import string
import csv


def generate_random_string(length=10):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def generate_dataset(size):
    return [generate_random_string() for _ in range(size)]


def create_datasets(size, intersection_ratio):
    initial_dataset = generate_dataset(size)

    intersection_size = int(size * intersection_ratio)

    intersection = random.sample(initial_dataset, intersection_size)

    dataset1 = intersection.copy()
    dataset1.extend(generate_dataset(size - intersection_size))
    dataset2 = intersection.copy()
    dataset2.extend(generate_dataset(size - intersection_size))

    return dataset1, dataset2


def write_dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in dataset:
            writer.writerow([item])


if __name__ == '__main__':
    size = 10
    intersection_ratio = 0.3
    dataset1, dataset2 = create_datasets(size, intersection_ratio)

    write_dataset_to_csv(dataset1, './dataset1.csv')
    write_dataset_to_csv(dataset2, './dataset2.csv')

    print(f"Dataset 1 size: {len(dataset1)}")
    print(f"Dataset 2 size: {len(dataset2)}")
