from typing import Type
from typing import Union

from torch import Tensor
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim import Optimizer


class amp(object):
    def __init__(self) -> None:
        super(amp, self).__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def scale(self, loss: Tensor) -> Tensor:
        raise NotImplementedError

    def unscale(self, optimizer: Optimizer) -> None:
        raise NotImplementedError

    def step(self, optimizer: Optimizer) -> None:
        raise NotImplementedError


class fp32(amp):
    def scale(self, loss: Tensor) -> Tensor:
        return loss

    def unscale(self, optimizer: Optimizer) -> None:
        pass

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()


class fp16(amp):
   