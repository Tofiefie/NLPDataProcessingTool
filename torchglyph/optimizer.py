from logging import getLogger
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

from torch import nn
from torch import optim

logger = getLogger(__name__)

ignores_default = (
    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,

    nn.SyncBatchNorm,
  