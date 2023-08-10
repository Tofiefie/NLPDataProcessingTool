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
from torch.distributions.utils impor