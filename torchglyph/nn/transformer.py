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
from torchglyph.nn.con