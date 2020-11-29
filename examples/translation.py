
# from dataclasses import dataclass
# from dataclasses import field
# from logging import getLogger
# from pathlib import Path
# from typing import Tuple
# from typing import Type
# from typing import Union
#
# import torch
# from torch import distributed as dist
# from torch import nn
# from torch.nn.utils.rnn import PackedSequence
# from torch.optim import Optimizer
# from torch.types import Number
# from torchrua import cat_padded_sequence
# from torchrua import CattedSequence
# from torchrua import pad_sequence
# from torchrua import roll_sequence
# from tqdm import tqdm
# from transformers import PreTrainedTokenizer
#
# from examples import project_out_dir
# from torchglyph import DEBUG
# from torchglyph import optimizer as optim
# from torchglyph import scheduler as sched
# from torchglyph.amp import Amp
# from torchglyph.amp import fp16
# from torchglyph.data.translation import Datasets
# from torchglyph.data.translation import wmt14deen
# from torchglyph.env import init_env
# from torchglyph.env import init_rank
# from torchglyph.meter import AccuracyMeter
# from torchglyph.meter import AverageMeter
# from torchglyph.meter import Meter
# from torchglyph.meter import SequenceMeter
# from torchglyph.nn.criterion import CrossEntropy
# from torchglyph.nn.embedding import TransformerEmbedding
# from torchglyph.nn.linear import Classifier
# from torchglyph.nn.mask import casual_mask
# from torchglyph.nn.mask import padding_mask
# from torchglyph.nn.transformer import TransformerDecoder
# from torchglyph.nn.transformer import TransformerEncoder
# from torchglyph.nn.utils import gather
# from torchglyph.nn.utils import mask_fill
# from torchglyph.nn.utils import sequence_like
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
# class Translator(nn.Module):
#     def __init__(self,
#                  max_length: int = 512, share_vocab: bool = False, tie_weight: bool = True,
#                  emb_: Type[TransformerEmbedding] = TransformerEmbedding,