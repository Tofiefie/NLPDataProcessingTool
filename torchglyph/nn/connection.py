
from typing import Type
from typing import Union

import torch
from torch import Tensor

from torchglyph.nn.normalization import LayerNorm


class PreLayerNorm(LayerNorm):
    def forward(self, tensor: Tensor, *, sub_layer, **kwargs):
        out = sub_layer(super(PreLayerNorm, self).forward(tensor), **kwargs)

        if torch.is_tensor(out):