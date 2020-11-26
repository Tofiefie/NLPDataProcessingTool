
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
#         h1, s1 = self(batch)
#         h2, s2 = self(batch)
#
#         task_loss1 = self.cky(h1).log_prob(targets=target).neg().sum() / batch_size2
#         task_loss2 = self.cky(h2).log_prob(targets=target).neg().sum() / batch_size2
#         task_loss = task_loss1 + task_loss2
#
#         token = batch["token"].data
#         mask = token[:, None] == token[None, :]
#         mask ^= torch.eye(mask.size()[0], dtype=torch.bool, device=mask.device)
#
#         s = torch.cosine_similarity(s1[:, None, :], s2[None, :, :], dim=-1) / self.tau
#         s[mask] = -float('inf')
#
#         hash_loss1 = torch.logsumexp(s, dim=-1).mean() - torch.diag(s).mean()
#         hash_loss2 = torch.logsumexp(s, dim=-2).mean() - torch.diag(s).mean()
#         hash_loss = hash_loss1 + hash_loss2
#
#         meter.task.update_by_mean(task_loss1, batch_size2)
#
#         return task_loss + self.beta * hash_loss
#
#     @torch.inference_mode()
#     def decode(self, batch, meter: InferenceMeter):
#         raise NotImplementedError
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
# class InverseSquareRootScheduler(sched.InverseSquareRootScheduler):
#     def __init__(self, num_training_steps: int = 5_0000, num_warmup_steps: int = 5000, *,
#                  optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
#         super(InverseSquareRootScheduler, self).__init__(
#             num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps,
#             optimizer=optimizer, last_epoch=last_epoch, **kwargs,
#         )
#
#
# def train_main(
#         rank: int, out_dir: Path, /,
#         setup_rank: Union[Type[init_rank]] = init_rank,
#         data,
#         model: Type[ConstituencyParsing] = ConstituencyParsing,
#         optimizer: Type[Adam] = Adam,
#         scheduler: Type[InverseSquareRootScheduler] = InverseSquareRootScheduler,
#         grad_norm: float = -1,
#         amp: Amp = fp16,
#         acc_interval: int = 1,
#         log_interval: int = 1 if DEBUG else 50,
#         dev_interval: int = 10 if DEBUG else 2000):
#     device = setup_rank(rank, out_dir)
#
#     train_loader, dev_loader, test_loader = data()
#
#     model = model().to(device=device)
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
#
#         if global_step % dev_interval == 0:
#             dev_meter = dev_stage(data_loader=dev_loader)
#             dev_meter.gather().log(stage='dev', iteration=global_step, out_dir=out_dir)
#
#             if dev_sota is None or dev_sota < dev_meter:
#                 dev_sota = dev_meter
#                 dev_sota.gather().log(stage='sota', iteration=global_step, out_dir=out_dir)
#
#         if global_step >= scheduler.num_training_steps:
#             break
#
#
# def train_constituency_parsing(setup_env: Type[init_env] = init_env, main: Type[train_main] = train_main, **kwargs):
#     out_dir = setup_env(project_out_dir=project_out_dir, **kwargs['@aku'])
#     device_count = torch.cuda.device_count()
#
#     if device_count == 1:
#         return main(-1, out_dir)
#
#     try:
#         torch.multiprocessing.spawn(
#             main, args=(out_dir,),
#             nprocs=device_count,
#         )
#     finally:
#         dist.destroy_process_group()