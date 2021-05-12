from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Tuple
from typing import Type

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from datasets.config import DATASETDICT_JSON_FILENAME
from tokenizers import Tokenizer
from torch.types import Device
from torchrua import cat_sequence

from torchglyph import data_dir
from torchglyph.data.abc import DataLoader
from torchglyph.data.abc import DataStore
from torchglyph.dist import get_device
from torchglyph.formats.ptb import iter_ptb
from torchglyph.io import all_exits
from torchglyph.io import cache_folder
from torchglyph.io import lock_folder
from torchglyph.nn.plm import RobertaBase
from torchglyph.nn.plm.abc import PLM
from torchglyph.tokenize_utils import encode_batch
from torchglyph.tokenize_utils import get_iterator
from torchglyph.tokenize_utils import train_word_tokenizer

WORD_FILENAME = 'word_tokenizer.json'
TARGET_FILENAME = 'target_tokenizer.json'


class ConstituencyParsing(DataStore):
    lang: str

    @classmethod
    def get_tokenize_fn(cls, plm: PLM, word_tokenizer: Tokenizer, target_tokenizer: Tokenizer, **kwargs):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            token, segment_size = plm.tokenize_batch(examples['word'], add_prefix_space=True)

            return {
                'word': encode_batch(examples['word'], tokenizer=wor