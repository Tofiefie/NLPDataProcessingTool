
import warnings
from typing import Iterator
from typing import List

import torch
from datasets import Dataset
from torch.utils import data as utils

from torchglyph.dist import get_rank


class RandomSampler(utils.RandomSampler):
    def __init__(self, dataset: Dataset, replacement: bool = False, num_samples: int = None) -> None:
        super(RandomSampler, self).__init__(data_source=dataset, replacement=replacement, num_samples=num_samples)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            self.generator = torch.default_generator

        yield from super(RandomSampler, self).__iter__()


class SequentialSampler(utils.SequentialSampler):
    def __init__(self, dataset: Dataset) -> None:
        super(SequentialSampler, self).__init__(data_source=dataset)


class SortishSampler(utils.Sampler[int]):
    def __init__(self, dataset: Dataset, section_size: int) -> None:
        super(SortishSampler, self).__init__(data_source=dataset)

        self.section_size = section_size
        self.sizes = dataset['size']

        self.last_indices = []
        self.descending = None

    def __len__(self) -> int:
        return len(self.sizes)

    def extend(self, last_indices: List[int]) -> None:
        self.last_indices.extend(last_indices)

    def __iter__(self) -> Iterator[int]:
        if self.descending is None:
            self.descending = get_rank() % 2 == 0

        sizes = torch.tensor(self.sizes, dtype=torch.long)
        randperm = torch.randperm(len(self), dtype=torch.long, generator=torch.default_generator)

        if len(self.last_indices) > 0: