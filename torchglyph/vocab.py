import itertools
from logging import getLogger
from typing import List
from typing import Tuple
from typing import Type

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from tokenizers.trainers import WordPieceTrainer
from torch.distributions.utils import lazy_property

logger = getLogger(__name__)

__all__ = [
    'WordVocab',
    'WordPieceVocab',
]


class Vocab(object):
    Token: Type
    Index: Type
    registry = {}

    def __init__(self, vocab_size: int = 10_0000, min_freq: int = 0, *,
                 unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 mask_token: str = None, special_tokens: Tuple[str, ...] = ()) -> None:
        super(Vocab, self).__init__()

        self.vocab_size = vocab_size
        self.min_freq = min_freq

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        special_tokens = [unk_token, pad_token, bos_token, eos_token, mask_token, *specia