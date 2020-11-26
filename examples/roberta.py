
# import math
# from typing import Optional
# from typing import Tuple
#
# import torch
# from torch import nn
# from transformers import AutoModel
# from transformers import PreTrainedModel
# from transformers import RobertaModel
#
# from torchglyph.nn.plm.abc import PLM
#
#
# class RobertaSelfAttention(nn.Module):
#     def __init__(self, config, num_bits: int, position_embedding_type=None) -> None:
#         super(RobertaSelfAttention, self).__init__()
#
#         self.num_attention_heads = num_bits
#         self.attention_head_size = (config.hidden_size + config.num_attention_heads - 1) // config.num_attention_heads
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.position_embedding_type = position_embedding_type or getattr(
#             config, "position_embedding_type", "absolute"
#         )
#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             self.max_position_embeddings = config.max_position_embeddings