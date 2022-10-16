
import datetime
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import get_type_hints

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Number
from torchrua import CattedSequence

from torchglyph.dist import all_gather_object
from torchglyph.dist import is_master
from torchglyph.io import save_sota

logger = getLogger(__name__)

TCP = Union[Tuple, CattedSequence, PackedSequence]


def detach_tensor(method):
    def wrap(self, *args: Union[Number, Tensor]):
        detached_args = []
        for arg in args:
            if torch.is_tensor(arg):
                arg = arg.detach().cpu().item()
            detached_args.append(arg)

        return method(self, *detached_args)

    return wrap


def zero_division(default):
    def wrap1(method):
        def wrap2(self, *args):
            try:
                return method(self, *args)
            except ZeroDivisionError:
                return default

        return wrap2

    return wrap1


class Meter(object):
    @property
    def keys(self) -> Tuple[Number, ...]:
        raise NotImplementedError

    def __eq__(self, other: 'Meter') -> bool:
        return self.keys == other.keys

    def __lt__(self, other: 'Meter') -> bool:
        return self.keys < other.keys

    def __gt__(self, other: 'Meter') -> bool:
        return self.keys > other.keys

    def __ne__(self, other: 'Meter') -> bool:
        return self.keys != other.keys

    def __le__(self, other: 'Meter') -> bool:
        return self.keys <= other.keys

    def __ge__(self, other: 'Meter') -> bool:
        return self.keys >= other.keys

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        out = {}

        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                for keys, value in getattr(self, name).stats.items():
                    out[(name, *keys)] = value

            else:
                logger.critical(f'field {name} is ignored')

        return out

    def log(self, stage: str, iteration: int, out_dir: Path = None):
        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                msg = ' | '.join(
                    f"{'-'.join([name, *keys])} {value}"
                    for keys, value in getattr(self, name).stats.items()
                )
                logger.info(f'{stage} {iteration} => {msg}')

                if is_master():
                    save_sota(out_dir=out_dir, step=iteration, **{
                        '-'.join([stage, name, *keys]): value
                        for keys, value in getattr(self, name).stats.items()
                    })
            else:
                logger.critical(f'field {name} is ignored')

        return self

    def gather(self):
        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                getattr(self, name).gather()
            else:
                logger.critical(f'field {name} is ignored')

        return self

    def update(self, *args) -> None:
        raise NotImplementedError


@dataclass()
class MaxMeter(Meter):
    value: Number = -float('inf')

    @property
    def max(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.max,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.max}

    @detach_tensor
    def update(self, value) -> None:
        self.value = max(self.value, value)

    def gather(self) -> None:
        self.value = max(all_gather_object(self.value))


@dataclass()
class MinMeter(Meter):
    value: Number = +float('inf')

    @property
    def min(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.min,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.min}

    @detach_tensor
    def update(self, value) -> None:
        self.value = min(self.value, value)

    def gather(self) -> None:
        self.value = min(all_gather_object(self.value))


@dataclass()
class SumMeter(Meter):
    value: Number = 0

    @property
    def sum(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.sum,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.sum}

    @detach_tensor
    def update(self, value) -> None:
        self.value += value

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))


@dataclass()
class AverageMeter(Meter):
    value: Number = 0
    weight: Number = 0

    @property
    @zero_division(default=0)
    def average(self) -> Number:
        return round(self.value / self.weight, ndigits=2)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.average,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.average}

    @detach_tensor
    def update_by_sum(self, value, weight=1) -> None:
        self.value += value
        self.weight += weight

    @detach_tensor
    def update_by_mean(self, value, weight=1) -> None:
        self.value += value * weight
        self.weight += weight

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))
        self.weight = sum(all_gather_object(self.weight))


@dataclass()