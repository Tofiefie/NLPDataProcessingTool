import torch
from torch import Tensor
from torch.nn import init
from torch.nn.init import calculate_gain
from torch.nn.init import constant_
from torch.nn.init import ones_
from torch.nn.init import orthogonal_
from torch.nn.init import zeros_

__all__ = [
    'constant_', 'zeros_', 'ones_',
    'xavier_normal_', 'kaiming_normal_',
    'xavier_uniform_', 'kaiming_uniform_',
    'bert_normal_', 'orthogonal_',
]


@torch.no_grad()
def xavier_normal_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.) -> Tensor:
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return init.normal_(tensor, mean=0, std=std)


@torch.no_grad()
def xavier_uniform_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.) -> Tensor:
    bound = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return init.uniform_(tensor, a=-bound, b=+bound)


@torch.no_grad()
def kaimi