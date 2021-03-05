from typing import Type
from typing import Union

from torch import Tensor
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim import Optimizer


class amp(object):
    def __init__