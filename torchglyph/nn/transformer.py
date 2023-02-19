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
            f'hidden_size={self.hidden_