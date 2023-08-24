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
                 mask_token: str = None, special_to