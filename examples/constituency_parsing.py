
# from dataclasses import field, dataclass
# from logging import getLogger
# from pathlib import Path
# from typing import Type
# from typing import Union
#
# import torch
# from tokenizers import Tokenizer
# from torch import distributed as dist
# from torch import nn
# from torch.optim import Optimizer
# from torchrua import major_sizes_to_ptr, major_sizes_to_size, trunc_sequence
# from torchrua import segment_sequence
# from tqdm import tqdm
# from transformers import RobertaTokenizer, RobertaConfig
# from transformers.models.roberta.modeling_roberta import RobertaModel
#
# from examples import project_out_dir
# from torchglyph import DEBUG
# from torchglyph import optimizer as optim
# from torchglyph import scheduler as sched
# from torchglyph.amp import fp16, Amp
# from torchglyph.dist import is_master
# from torchglyph.env import init_rank, init_env
# from torchglyph.data.abc import DataLoader
# from torchglyph.meter import Meter, SequenceMeter, AverageMeter, AccuracyMeter
# from torchglyph.nn.plm.abc import PLM
# from torchglyph.nn.plm.tuning import qof
# from torchlatent.cky import CkyDecoder
#
# logger = getLogger(__name__)
#
#
# @dataclass
# class FitMeter(Meter):
#     token: SequenceMeter = field(default_factory=SequenceMeter)
#     task: AverageMeter = field(default_factory=AverageMeter)
#     hash: AverageMeter = field(default_factory=AverageMeter)
#
#
# @dataclass
# class InferenceMeter(Meter):
#     src: SequenceMeter = field(default_factory=SequenceMeter)
#     acc: AccuracyMeter = field(default_factory=AccuracyMeter)
#
#
# class ConstituencyParsing(nn.Module):
#     def __init__(self, tuning: Type[qof] = qof, hidden_features: int = 128, dropout: float = 0.3,
#                  tau: float = 0.1, beta: float = 0.005, *,
#                  plm: PLM, target_tokenizer: Tokenizer) -> None:
#         super(ConstituencyParsing, self).__init__()
#
#         self.tau = tau
#         self.beta = beta
#         self.config: RobertaConfig = plm.config
#         self.tokenizer: RobertaTokenizer = plm.tokenizer
#         self.model: RobertaModel = plm.model
#         tuning(self=self.model)
#         self.model.encoder.layer[-1].requires_grad_(True)
#
#         self.cky = CkyDecoder(
#             in_features=self.config.hidden_size,
#             hidden_features=hidden_features,
#             num_targets=target_tokenizer.get_vocab_size(),
#             dropout=dropout,
#         )
#
#     def forward(self, batch):
#         device = batch["token"].data.device
#
#         t, b = major_sizes_to_size(sizes=batch["token"].token_sizes)
#         token_ptr, batch_ptr = major_sizes_to_ptr(sizes=batch["token"].token_sizes)
#
#         input_ids = torch.full((b, t), dtype=torch.long, device=device, fill_value=self.tokenizer.pad_token_id)
#         attention_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
#
#         input_ids[batch_ptr, token_ptr] = batch["token"].data
#         attention_mask[batch_ptr, token_ptr] = True
#
#         out = self.model.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_attentions=True,
#             return_dict=True,
#         )
#
#         sequence = segment_sequence(batch["segment_size"], out.last_hidden_state, reduce='mean', batch_first=True)
#         attention = out.attentions[-1][batch_ptr, :, token_ptr, token_ptr]
#
#         return trunc_sequence(sequence, trunc=(1, 1)), attention
#
#     def fit(self, batch, meter: FitMeter):
#         data = [batch["start"].data, batch["end"].data, batch["target"].data]
#         target = batch["target"]._replace(data=torch.stack(data, dim=-1))
#
#         t, b = major_sizes_to_size(sizes=batch["segment_size"].token_sizes)
#         batch_size1 = batch["segment_size"].token_sizes.size()[0]
#         batch_size2 = batch_size1 - b * 2
#