
# from dataclasses import dataclass
# from dataclasses import field
# from logging import getLogger
# from pathlib import Path
# from typing import Tuple
# from typing import Type
# from typing import Union
#
# import torch
# from tokenizers import Tokenizer
# from torch import distributed as dist
# from torch import nn
# from torch import optim
# from torch.optim import Optimizer
# from torch.types import Number
# from torchlatent.crf import CrfDecoder
# from torchlatent.crf import CrfDistribution
# from torchrua.segment import segment_sequence
# from tqdm import tqdm
# from transformers import RobertaConfig
# from transformers import RobertaModel
# from transformers import RobertaTokenizer
#
# from examples import project_out_dir
# from torchglyph import DEBUG
# from torchglyph import optimizer as optim
# from torchglyph import scheduler as sched
# from torchglyph.amp import Amp
# from torchglyph.amp import fp16
# from torchglyph.data.abc import DataLoader
# from torchglyph.data.token_classification import conll2003
# from torchglyph.data.token_classification import Data
# from torchglyph.dist import is_master
# from torchglyph.env import init_env
# from torchglyph.env import init_rank
# from torchglyph.meter import AccuracyMeter
# from torchglyph.meter import AverageMeter
# from torchglyph.meter import Meter
# from torchglyph.meter import SequenceMeter
# from torchglyph.nn.mask import prepare_input_ids
# from torchglyph.nn.plm import PLM
#
# logger = getLogger(__name__)
#
#
# @dataclass
# class FitMeter(Meter):
#     src: SequenceMeter = field(default_factory=SequenceMeter)
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
#     acc: AccuracyMeter = field(default_factory=AccuracyMeter)
#
#
# class Tagger(nn.Module):
#     def __init__(self, dropout: float = 0.5, *, plm: PLM, target_tokenizer: Tokenizer) -> None:
#         super(Tagger, self).__init__()
#
#         self.plm = plm
#         self.config: RobertaConfig = plm.config
#         self.tokenizer: RobertaTokenizer = plm.tokenizer
#         self.model: RobertaModel = plm.model
#
#         self.pad_token_id = self.tokenizer.pad_token_id
#
#         self.crf = CrfDecoder(
#             in_features=self.config.hidden_size,
#             num_targets=target_tokenizer.get_vocab_size(),
#             num_conjugates=1,
#             dropout=dropout,
#         )
#
#     def forward(self, batch) -> CrfDistribution:
#         input_ids, attention_mask = prepare_input_ids(
#             batch["token"], padding_id=self.pad_token_id,
#             device=batch["token"].data.device,
#         )
#
#         out = self.model.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True,
#         )
#
#         sequence = segment_sequence(
#             tensor=out.last_hidden_state, sizes=batch["segment_size"],
#             reduce='mean', batch_first=True,
#         )
#
#         return self.crf.forward(sequence=sequence)
#
#     def fit(self, batch, meter: FitMeter):
#         dist: CrfDistribution = self(batch)
#
#         batch_size, *_ = batch["target"].data.size()
#         loss = (dist.log_partitions - dist.log_scores(targets=batch["target"])).sum()
#
#         meter.nll.update_by_sum(loss, batch_size)
#
#         return loss / batch_size
#
#     @torch.inference_mode()
#     def decode(self, batch, meter: InferenceMeter):
#         dist: CrfDistribution = self(batch)
#
#         return batch["segment_size"]._replace(data=dist.argmax)
#
#
# class Adam(optim.Adam):
#     def __init__(self, lr: float = 7e-4, beta1: float = 0.9, beta2: float = 0.98,
#                  weight_decay: float = 1e-4, amsgrad: bool = False, *, model: nn.Module, **kwargs) -> None:
#         params_with_decay, params_without_decay = optim.divide_groups(model)