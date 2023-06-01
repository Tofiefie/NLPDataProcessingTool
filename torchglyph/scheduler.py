
from logging import getLogger
from typing import Type
from typing import Union

from torch.optim import Optimizer
from torch.optim import lr_scheduler

logger = getLogger(__name__)


class LambdaLR(lr_scheduler.LambdaLR):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def step(self, epoch: int = None) -> None:
        super(LambdaLR, self).step(epoch=epoch)

    def report_lr(self) -> None:
        for group, lr in enumerate(self.get_last_lr()):
            logger.info(f'group {group} | lr => {lr:.10f}')


class ConstantScheduler(LambdaLR):
    def __init__(self, num_training_steps: int = 20_0000, num_warmup_steps: int = 5000, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps