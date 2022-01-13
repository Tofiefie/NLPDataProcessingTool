import os
import random
from logging import getLogger
from socket import socket
from typing import Any
from typing import List

import numpy as np
import torch
from torch import Generator
from torch import Tensor
from torch import distributed

logger = getLogger(__name__)


def get_port() -> int:  # TODO: resolve this
    sock = socket()
    sock.bind(('', 0))

    _, port = sock.getsockname()
    return port


def init_process(*, rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'
    distributed.init_process_group(
        backend='nccl', init_method=f'env://',
        world_size=torch.cuda.device_count(), rank=rank,
    )

    torch.cuda.set_device(rank)
    torch.cuda.synchronize(rank)


def init_seed(seed: int = 42, *, rank: int) -> None:
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return logger.warning(f'{rank}.seed <- {seed}')


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if not distributed.is_initialized():
        return torch.device('cuda:0')

    return torch.device(f'cuda:{distributed.get_rank()}')


def get_generator() -> Generator:
    if not torch.cuda.is_available():
        return torch.default_generator

    if not distributed.is_initialized():
        return torch.cuda.default_generators[0]

    return torch.cuda.default_generators[distributed.get_rank()]


def get_rank() -> int:
    if not distributed.is_initialized():
        return 0

    return distributed.get_rank()


def get_world_size() -> int:
    if not distributed.is_initialized():
        return 1

    return distributed.get_world_size()


def is_master() -> bool:
    if not distributed.is_