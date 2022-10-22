from typing import Type
from typing import Union

from torch import nn


class ReLU(nn.ReLU):
    def __init__(self) -> None:
        super(ReL