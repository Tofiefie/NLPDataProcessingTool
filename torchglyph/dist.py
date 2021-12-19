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


de