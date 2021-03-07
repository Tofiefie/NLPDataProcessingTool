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
        raise NotImplemente