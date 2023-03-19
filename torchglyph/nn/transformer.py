import itertools
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

from torch import Tensor
from torch import nn

from torchglyph.nn.activation import Activations
from torchglyph.nn.activation import ReLU
from torchglyph.nn.attention import Cache
from torchglyph.nn.attention import CrossAttention
from torchglyph.nn.attention import SelfAttention
from torchglyph.nn.connection import Connections
from torchglyph.nn.connection import PostLayerNorm
from torchglyph.nn.utils import gather


class TransformerFfn(nn.Sequential):
    def __init__(self, bias: bool = True, activation: Activations = ReLU, *,
                 in_features: int, dropout: float) -> None:
        self.in_features = in_features
        self.hidden_features = in_features * 4
        self.out_features = in_features

        super(TransformerFfn, self).__init__(
            nn.Linear(self.in_features, self.hidden_features, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_features, self.out_features, bias=bias),
            nn.Dropout(dropout),
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_size={self.in_features}',
            f'hidden_size={self.hidden_features}',
            f'dropout={self[-1].p}',
        ])


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 self_: Type[SelfAttention] = SelfAttention,
                 ffn_: Type[TransformerFfn] = TransformerFfn,
                 layer_norm_: Connections = PostLayerNorm,
                 dropout: float = 0.1, *, in_size: int) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.self = self_(q_dim=in_size, o_dim=in_size, dropout=dropout)

        self.ffn = ffn_(in_features=in_size, dropout=dropout)
        self.encoding_dim = self.ffn.out_features

        self.dropout = nn.Dropout(dropout)
        self.norm1 = layer_norm_(in_size=in_size)
        self.norm2 = layer_norm_(in_size=in_size)

    def att(self, tensor: Tensor, mask: Tensor) -> Tensor:
        tensor, _, _ = self.self(tensor, mask=mask)
        return self.dropout(tensor)

    def forward(self, tensor: Tensor, mask: Tensor = None) -> Tensor:
        tensor = self.norm1(tensor, sub_layer=self.att, mask=mask)
        tensor = self.norm2(tensor, sub_layer=self.ffn)
        return tensor


class TransformerEncoder(nn.ModuleList):
    def __init__(self, layer: Type[TransformerEncoderLayer] = TransformerEncoderLayer,
                 num_layers: int = 6, *, in_size: int) -> None:
        modules = []
        for _ in range(num_layers):
            modules.append(layer(in_size=in_size))
            in_size = modules[-1].encoding_dim

        super(TransformerEncoder, self).__init__(modules)
        self.encoding_dim = modules[-1].encoding_dim

    def forward(self, tensor: Tensor, mask: Tensor = None) -> Tensor:
        for layer in self:
            tensor = layer(tensor=tensor, mask=mask)
        return tensor


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 self_: Type[SelfAttention] = SelfAttention,
                 cross_: Type[CrossAttention] = CrossAttention,
                 ffn_: Type[TransformerFfn] = TransformerFfn,
                 layer_norm_: Connections = PostLayerNorm,
                 dropout: float = 0.1, *, in_size: int) -> None:
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention = self_(q_dim=in_size, o_dim=in_size, dropout=dropout)
        self.cross_attention = cross_(q_dim=in_size, kv_dim=in_size, o_dim=in_size, dropout=dropout)

        self.ffn = ffn_(in_features=in_size, dropout=dropout)
        self.encoding_dim = self.ffn.out_features

        self.dropout = nn.Dropout(dropout)
        self.norm1 = layer_norm_(in_size=in_size)
        self.norm2 = layer_norm_(in_size=in_size)
        self.norm3 = layer_norm_(in_size=in_siz