import torch
from torch import Tensor
from torch.nn import init
from torch.nn.init import calculate_gain
from torch.nn.init import constant_
from torch.nn.init import ones_
from torch.nn.init import orthogonal_
from torch.nn.init import zeros_

__all__ = [
    'constant_', 'zer