from logging import getLogger
from typing import Tuple

import torch
from tokenizers import Tokenizer
from torch import Tensor

logger = getLogger(__name__)


def align_tokenizer(tokenizer: Tokenizer, pretrained_tokenizer: Tokenizer, *transforms) -> Tuple[Tensor, Tensor]:
    count, xs, ys = 0, [], []

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    pretrained_vocab = pretrained_tokenizer.get_vocab(with_added_tokens=True)

    for token, index in tokenizer.get_vocab(with_added_tokens=True).items():
        for transform in (str, *transforms):
            pretrained_token = transform(token)
            pretrained_index = pretrained_vocab.get(pretrained_token, None)

            if pretrained_token is not None:
                xs.append(index)
                ys.append(pretrained_index)
                count += 1
             