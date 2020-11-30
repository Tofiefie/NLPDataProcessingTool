
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
#                  enc_: Type[TransformerEncoder] = TransformerEncoder,
#                  dec_: Type[TransformerDecoder] = TransformerDecoder,
#                  classifier_: Type[Classifier] = Classifier,
#                  criterion: Type[CrossEntropy] = CrossEntropy,
#                  *,
#                  src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer):
#         super(Translator, self).__init__()
#
#         self.bos_index = tgt_tokenizer.bos_token_id
#         self.eos_index = tgt_tokenizer.eos_token_id
#         self.pad_index = tgt_tokenizer.pad_token_id
#         logger.info(f'self.bos_index => {self.bos_index}')
#         logger.info(f'self.eos_index => {self.eos_index}')
#         logger.info(f'self.pad_index => {self.pad_index}')
#
#         self.src_tokenizer = src_tokenizer
#         self.tgt_tokenizer = tgt_tokenizer
#
#         self.src_embed = self.tgt_embed = emb_(
#             num_embeddings=src_tokenizer.vocab_size,
#             max_length=max_length,
#             padding_idx=src_tokenizer.pad_token_id,
#         )
#
#         if not share_vocab:
#             self.tgt_embed = emb_(
#                 num_embeddings=tgt_tokenizer.vocab_size,
#                 max_length=max_length,
#                 padding_idx=tgt_tokenizer.pad_token_id,
#             )
#
#         self.encoder = enc_(in_size=self.src_embed.embedding_dim)
#         self.decoder = dec_(in_size=self.encoder.encoding_dim)
#
#         self.classifier = classifier_(
#             in_features=self.decoder.encoding_dim,
#             out_features=tgt_tokenizer.vocab_size,
#             padding_idx=tgt_tokenizer.pad_token_id,
#         )
#         if tie_weight:
#             self.classifier.tie_parameter(weight=self.tgt_embed.weight)
#
#         self.criterion = criterion(ignore_index=tgt_tokenizer.pad_token_id)
#
#     def forward_src(self, src: Union[CattedSequence, PackedSequence]):
#         mask = padding_mask(src)  # [b, ..., 1, s]
#         embedding = self.src_embed(src)  # [b, ..., s, d]
#         encoding = self.encoder(tensor=embedding, mask=mask)  # [b, ..., s, d]
#         return encoding, mask
#
#     def forward(self, batch, meter: FitMeter):
#         memory, memory_mask = self.forward_src(batch["src"])
#         mask = casual_mask(batch["tgt"])  # [b, ..., t, t]
#         tensor = self.tgt_embed(roll_sequence(batch["tgt"], shifts=1))  # [b, ..., t, d]
#
#         tgt_encoding, _, _ = self.decoder.forward(  # [b, ..., t, d]
#             tensor=tensor, mask=mask,
#             memory=memory, memory_mask=memory_mask,
#         )
#
#         logits = self.classifier(tgt_encoding)  # [b, ..., t, n]
#
#         padded_tgt, _ = pad_sequence(batch["tgt"], batch_first=True, padding_value=self.pad_index)
#         loss = self.criterion(logits, target=padded_tgt)
#
#         prd = sequence_like(logits.argmax(dim=-1), sequence=batch["tgt"])
#
#         meter.src.update(batch["src"])
#         meter.tgt.update(batch["tgt"])
#         meter.nll.update_by_sum(loss, 1)
#         meter.acc.update_by_sequence(prd, batch["tgt"])
#
#         return loss
#
#     @torch.inference_mode()
#     def decode(self, batch, meter: InferenceMeter, k: int = 5,
#                min_length: int = 1, max_length: int = 256, length_penalty: float = 1):
#         device = batch["src"].data.device
#
#         memory, memory_mask = self.forward_src(batch["src"])  # [b, s, d], [b, 1, s]
#         memory, memory_mask = memory[:, None], memory_mask[:, None]  # [b, 1, s, d], [b, 1, 1, s]
#
#         b, *_ = memory.size()
#         index0 = torch.arange(b, device=device, dtype=torch.long)[..., None]
#
#         done = torch.zeros((b, k), dtype=torch.bool, device=device)
#         lengths = torch.ones((b, k), dtype=torch.float32, device=device)
#         scores = torch.zeros((b, k), dtype=torch.float32, device=device)
#
#         tokens = torch.zeros((b, k, 2), dtype=torch.long, device=device)
#         tokens[..., 0] = self.eos_index
#         tokens[..., 1] = self.bos_index
#
#         tensor = self.tgt_embed(tokens[..., :1], position=0)  # [b, k, 1, d]
#         _, att_cache, crs_cache = self.decoder.forward(  # [b, k, 1, d]
#             tensor=tensor, mask=None,
#             memory=memory, memory_mask=memory_mask,
#             att_cache=None, crs_cache=None,
#         )
#
#         forbidden_tokens = [self.bos_index, self.eos_index]
#
#         for step in range(1, max_length):
#             if step == min_length:
#                 forbidden_tokens.remove(self.eos_index)
#
#             tensor = self.tgt_embed(tokens[..., -1:], position=step)  # [b, k, 1, d]
#             tensor, att_cache, crs_cache = self.decoder.forward(  # [b, k, 1, d]
#                 tensor=tensor, mask=None,
#                 memory=memory, memory_mask=memory_mask,
#                 att_cache=att_cache, crs_cache=crs_cache,
#             )
#
#             transition = self.classifier(tensor[..., -1, :])  # [b, k, vocab]
#             transition = mask_fill(transition, *forbidden_tokens)
#
#             transition, next_token = self.classifier.topk(transition, k=k)  # [b, k, l]
#             transition[done[..., None].expand_as(transition)] = 0
#
#             scores = scores[..., None] + transition
#             normed_scores = scores * (lengths[..., None] ** -length_penalty)
#             normed_scores = normed_scores.flatten(start_dim=-2)  # [b, kl]
#             _, index = torch.topk(normed_scores, k=k, dim=-1)  # [b, k]
#
#             index1, index2 = index // next_token.size()[-1], index % next_token.size()[-1]
#
#             att_cache = self.decoder.gather(caches=att_cache, dim=1, index=index1[..., None, None, None])
#
#             tokens = gather(tensor=tokens, dim=1, index=index1[..., None])  # [b, k, t]
#             tokens = torch.cat([tokens, next_token[index0, index1, index2, None]], dim=-1)  # [b, k, t+1]
#             scores = scores[index0, index1, index2]
#
#             done = done[index0, index1] | (tokens[..., -1] == self.eos_index)
#             lengths = lengths[index0, index1] + (~done).float()
#             if done.all().item():
#                 break
#
#         scores, index = torch.max(scores, dim=1, keepdim=False)
#         lengths, _ = torch.max(lengths, dim=1, keepdim=False)
#         tokens = gather(tokens, dim=1, index=index[:, None, None])
#
#         prd = cat_padded_sequence(tokens[:, 0, 1:], token_sizes=lengths - 1)
#
#         meter.src.update(batch["src"])
#         meter.tgt.update(batch["tgt"])
#         meter.prd.update(prd)
#         meter.acc.update_by_sequence(prd, batch["tgt"])
#
#         return prd
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
#     def __init__(self, num_training_steps: int = 20_0000, num_warmup_steps: int = 5000, *,
#                  optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
#         super(InverseSquareRootScheduler, self).__init__(
#             num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps,
#             optimizer=optimizer, last_epoch=last_epoch, **kwargs,
#         )
#
#
# def train_translator_rank(
#         rank: int, out_dir: Path, /,
#         setup_rank: Union[Type[init_rank]] = init_rank,
#         data: Datasets = wmt14deen,
#         model: Type[Translator] = Translator,
#         optimizer: Type[Adam] = Adam,
#         scheduler: Type[InverseSquareRootScheduler] = InverseSquareRootScheduler,
#         grad_norm: float = -1,
#         amp: Amp = fp16,
#         use_compile: bool = True,
#         acc_interval: int = 1,
#         log_interval: int = 1 if DEBUG else 50,
#         dev_interval: int = 10 if DEBUG else 2000):
#     device = setup_rank(rank, out_dir)
#
#     (train_loader, dev_loader, test_loader), (src_plm, tgt_plm) = data()
#
#     model = model(
#         src_tokenizer=src_plm.tokenizer,
#         tgt_tokenizer=tgt_plm.tokenizer,
#     ).to(device=device)
#     if dist.is_initialized():
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
#     if use_compile:
#         model = torch.compile(model=model, fullgraph=True)
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
#     model.train()
#     train_meter, sota = FitMeter(), None
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
#         if global_step >= scheduler.num_training_steps:
#             break
#
#
# def train_translator(setup_env: Type[init_env] = init_env,
#                      train_fn: Type[train_translator_rank] = train_translator_rank, **kwargs):
#     out_dir = setup_env(project_out_dir=project_out_dir, **kwargs['@aku'])
#
#     try:
#         torch.multiprocessing.spawn(
#             train_fn, args=(out_dir,),
#             nprocs=torch.cuda.device_count(),
#         )
#     finally:
#         dist.destroy_process_group()