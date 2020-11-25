
# from dataclasses import dataclass
# from dataclasses import field
# from logging import getLogger
# from pathlib import Path
# from typing import Literal
# from typing import NewType
# from typing import Tuple
# from typing import Type
# from typing import Union
#
# import torch
# from torch import distributed as dist
# from torch import nn
# from torch.optim import Optimizer
# from torch.types import Number
# from tqdm import tqdm
# from transformers import AutoModelForSeq2SeqLM
# from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
#
# from examples import project_out_dir
# from torchglyph import DEBUG
# from torchglyph import optimizer as optim
# from torchglyph import scheduler as sched
# from torchglyph.amp import Amp
# from torchglyph.amp import fp16
# from torchglyph.data import translation as sources
# from torchglyph.data.abc import DataLoader
# from torchglyph.dist import is_master
# from torchglyph.env import init_env
# from torchglyph.env import init_rank
# from torchglyph.meter import AccuracyMeter
# from torchglyph.meter import AverageMeter
# from torchglyph.meter import Meter
# from torchglyph.meter import SequenceMeter
# from torchglyph.nn import criterion
# from torchglyph.nn.mask import prepare_input_ids
# from torchglyph.nn.plm import MBartLarge
# from torchglyph.nn.plm import PLM
# from torchglyph.nn.plm.tuning import full
# from torchglyph.nn.plm.tuning import qof
#
# logger = getLogger(__name__)
#
#
# @dataclass
# class FitMeter(Meter):
#     src: SequenceMeter = field(default_factory=SequenceMeter)
#     tgt: SequenceMeter = field(default_factory=SequenceMeter)
#     nll: AverageMeter = field(default_factory=AverageMeter)
#     acc: AccuracyMeter = field(default_factory=AccuracyMeter)
#
#     @property
#     def keys(self) -> Tuple[Number, ...]:
#         return self.acc.keys
#
#
# @dataclass
# class InferenceMeter(Meter):
#     src: SequenceMeter = field(default_factory=SequenceMeter)
#     tgt: SequenceMeter = field(default_factory=SequenceMeter)
#     prd: SequenceMeter = field(default_factory=SequenceMeter)
#     acc: AccuracyMeter = field(default_factory=AccuracyMeter)
#
#
# class CrossEntropy(criterion.CrossEntropy):
#     def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean',
#                  label_smoothing: float = 0, *, ignore_index: int = -100) -> None:
#         super(CrossEntropy, self).__init__(reduction, label_smoothing, ignore_index=ignore_index)
#
#
# class BartTranslator(nn.Module):
#     def __init__(self, criterion: Type[CrossEntropy] = CrossEntropy,
#                  tuning: Union[Type[full], Type[qof]] = qof, *, plm: PLM) -> None:
#         super(BartTranslator, self).__init__()
#
#         self.pad_token_id = plm.tokenizer.pad_token_id
#
#         mbart: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(plm.pretrained_model_name)
#
#         self.model = mbart.model
#         tuning(self=self.model)
#
#         self.head = mbart.lm_head
#
#         self.criterion = criterion(ignore_index=self.pad_token_id)
#
#     def forward(self, batch, meter: FitMeter):
#         input_ids, attention_mask = prepare_input_ids(batch["src"], padding_id=self.pad_token_id)
#
#         labels, _ = prepare_input_ids(batch["tgt"], padding_id=self.pad_token_id)
#
#         rolled_tgt = batch["tgt"].roll(shifts=1)
#         decoder_input_ids, decoder_attention_mask = prepare_input_ids(rolled_tgt, padding_id=self.pad_token_id)
#
#         out = self.model.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             return_dict=True,
#         )
#         logits = self.head(out.last_hidden_state)
#
#         loss = self.criterion(logits, labels)
#         argmax = logits.argmax(dim=-1)
#
#         meter.src.update(batch["src"])
#         meter.tgt.update(batch["tgt"])
#         meter.nll.update_by_sum(loss, 1)
#         meter.acc.update_by_tensor(argmax, labels, pad_token_id=self.pad_token_id)
#
#         return loss
#
#
# class Adam(optim.Adam):
#     def __init__(self, lr: float = 3e-5, beta1: float = 0.9, beta2: float = 0.999,
#                  weight_decay: float = 1e-4, amsgrad: bool = False, *, model: nn.Module, **kwargs) -> None:
#         params_with_decay, params_without_decay = optim.divide_groups(model)
#         logger.info(f'len(params_with_decay) => {len(params_with_decay)}')
#         logger.info(f'len(params_without_decay) => {len(params_without_decay)}')
#         super(Adam, self).__init__(
#             lr=lr, beta1=beta1, beta2=beta2, amsgrad=amsgrad, params=[
#                 {'params': params_with_decay, 'weight_decay': weight_decay},
#                 {'params': params_without_decay, 'weight_decay': 0.},
#             ]
#         )
#
#
# class InverseSquareRootScheduler(sched.InverseSquareRootScheduler):
#     def __init__(self, num_training_steps: int = 10_0000, num_warmup_steps: int = 4000, *,
#                  optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
#         super(InverseSquareRootScheduler, self).__init__(
#             num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps,
#             optimizer=optimizer, last_epoch=last_epoch, **kwargs,
#         )
#
#
# class Wmt14DeEn(sources.Wmt14DeEn):
#     @classmethod
#     def new(cls, batch_size: int = 1024,
#             src_plm: Type[MBartLarge] = MBartLarge,
#             tgt_plm: Type[MBartLarge] = MBartLarge):
#         return super(Wmt14DeEn, cls).new(batch_size=batch_size, src_plm=src_plm, tgt_plm=tgt_plm)
#
#