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
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,

    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.InstanceNorm3d,
    nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.LazyInstanceNorm3d,
)


def divide_groups(module: nn.Module, ignores: Tuple[nn.Module, ...] = None):
    if ignores is None:
        ignores = ignores_default

    memory = set()
    with_decay = set()
    without_decay = set()

    def recur(mod: nn.Module):
        if mod in memory:
            return

        memory.add(mod)

        for name, param in mod.named_parameters(recurse=False):
            if param.requires_grad:
                if isinstance(mod, ignores) or 'bias' in name:
                    without_decay.add(param)
                else:
                    with_decay.add(param)

        for m in mod._modules.values():
            recur(mod=m)

    recur(mod=module)
    validate_groups(module, with_decay=with_decay, without_decay=without_