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

    _, port =