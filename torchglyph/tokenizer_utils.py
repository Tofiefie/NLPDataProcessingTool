from logging import getLogger
from typing import Tuple

import torch
from tokenizers import Tokenizer
from torch import Tensor

logger = getLogger(__name__)


def align_tokenizer(tokenizer: Tokenizer, pretrained_tokenizer: Tokenizer, *transforms) -> Tuple[Tensor, Tensor]:
    count, xs, ys = 0, [], []

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    pretrained_vocab = pr