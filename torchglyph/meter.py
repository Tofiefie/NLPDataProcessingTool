
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
