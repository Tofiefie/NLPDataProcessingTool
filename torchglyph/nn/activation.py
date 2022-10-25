from typing import Type
from typing import Union

from torch import nn


class ReLU(nn.ReLU):
    def __init__(self) -> None:
        super(ReLU, self).__init__()


class GELU(nn.GELU):
    def __init__(self) -> None:
        su