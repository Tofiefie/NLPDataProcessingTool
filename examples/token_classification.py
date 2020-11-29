
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
#         super(Adam, self).__init__(
#             lr=lr, beta1=beta1, beta2=beta2, amsgrad=amsgrad, params=[
#                 {'params': params_with_decay, 'weight_decay': weight_decay},
#                 {'params': params_without_decay, 'weight_decay': 0.},
#             ]
#         )
#
#
# class LinearScheduler(sched.LinearScheduler):
#     def __init__(self, num_training_steps: int = 5_0000, num_warmup_steps: int = 5000, *,
#                  optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
#         super(LinearScheduler, self).__init__(
#             num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps,
#             optimizer=optimizer, last_epoch=last_epoch, **kwargs,
#         )
#
#
# def train_main(
#         rank: int, out_dir: Path, /,
#         setup_rank: Union[Type[init_rank]] = init_rank,
#         data: Data = conll2003,
#         model: Type[Tagger] = Tagger,
#         optimizer: Type[Adam] = Adam,
#         scheduler: Type[LinearScheduler] = LinearScheduler,
#         grad_norm: float = 5,
#         amp: Amp = fp16,
#         acc_interval: int = 1,
#         log_interval: int = 1 if DEBUG else 50,
#         dev_interval: int = 10 if DEBUG else 2000):
#     device = setup_rank(rank, out_dir)
#
#     (train_loader, dev_loader, test_loader), (plm, target_tokenizer) = data()
#
#     model = model(plm=plm, target_tokenizer=target_tokenizer).to(device=device)
#     if dist.is_initialized():
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
#
#     logger.info(f'model => {model}')
#
#     optimizer = optimizer(model=model)
#     logger.info(f'optimizer => {optimizer}')
#
#     scheduler = scheduler(optimizer=optimizer)
#     logger.info(f'scheduler => {scheduler}')
#
#     amp = amp()
#     logger.info(f'amp => {amp}')
#
#     def dev_stage(data_loader: DataLoader):
#         meter = FitMeter()
#         model.eval()
#
#         desc = None
#         if is_master():
#             desc = tqdm(total=data_loader.dataset.num_rows, desc='dev')
#
#         for dev_batch in data_loader:
#             model(dev_batch, meter=meter)
#             if desc is not None:
#                 desc.update(dev_batch['batch_size'])
#
#         model.train()
#         return meter
#
#     model.train()
#     train_meter, dev_sota = FitMeter(), None
#     for global_step, batch in tqdm(enumerate(train_loader, start=1), desc=f'train', total=scheduler.num_training_steps):
#         with amp:
#             loss = model(batch, meter=train_meter) / acc_interval
#             if dist.is_initialized():
#                 dist.all_reduce(loss)
#         amp.scale(loss).backward()
#
#         if grad_norm > 0:
#             amp.unscale(optimizer=optimizer)
#             torch.nn.utils.clip_grad_norm_(
#                 parameters=model.parameters(),
#                 max_norm=grad_norm,
#             )
#
#         amp.step(optimizer=optimizer)
#         scheduler.step()
#
#         if global_step % log_interval == 0:
#             train_meter.gather().log(stage='train', iteration=global_step, out_dir=out_dir)
#             train_meter = FitMeter()