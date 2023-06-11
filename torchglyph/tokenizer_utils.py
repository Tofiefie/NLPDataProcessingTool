from logging import getLogger
from typing import Tuple

import torch
from tokenizers import Tokenizer
from torch import Tensor

logger = getLogger(__name__)


def align_tokenizer(tokenizer: Tokenizer, pr