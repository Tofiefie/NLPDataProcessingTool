
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import NewType
from typing import Tuple
from typing import Type
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from seqeval.metrics.sequence_labeling import get_entities
from tokenizers import Tokenizer
from torch.types import Device
from torchrua import cat_sequence

from torchglyph import data_dir
from torchglyph.data.abc import DataLoader
from torchglyph.data.abc import DataStore
from torchglyph.dist import get_device
from torchglyph.formats.conll import iter_sentence
from torchglyph.io import cache_folder
from torchglyph.io import is_dataset_dict_folder
from torchglyph.io import lock_folder
from torchglyph.nn.plm import PLM
from torchglyph.nn.plm import RobertaBase
from torchglyph.tokenize_utils import encode_batch
from torchglyph.tokenize_utils import get_iterator
from torchglyph.tokenize_utils import train_word_tokenizer


def convert_scheme(tags: List[str]) -> List[str]:
    out = ['O' for _ in tags]

    for name, x, y in get_entities(tags):
        if x == y:
            out[x] = f'S-{name}'
        else:
            out[x] = f'B-{name}'
            out[y] = f'E-{name}'
            for index in range(x + 1, y):
                out[index] = f'I-{name}'

    return out


class TokenClassification(DataStore):
    lang: str

    @classmethod
    def get_tokenize_fn(cls, plm: PLM, target_tokenizer: Tokenizer, **kwargs):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            token, segment_size = plm.tokenize_batch(examples['token'], add_prefix_space=True)

            return {
                'token': token,
                'segment_size': segment_size,
                'target': encode_batch(examples['target'], tokenizer=target_tokenizer),
                'size': [len(example) for example in examples['target']],
            }

        return tokenize

    @classmethod
    def predicate(cls, example) -> bool:
        return True

    @classmethod
    def get_collate_fn(cls, device: Device, **kwargs):
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                'token': cat_sequence([example['token'] for example in examples]).to(device=device),
                'segment_size': cat_sequence([example['segment_size'] for example in examples]).to(device=device),
                'target': cat_sequence([example['target'] for example in examples]).to(device=device),
            }

        return collate_fn

    @classmethod
    def load_split(cls, path: Path, **kwargs):
        with path.open(mode='r', encoding='utf-8') as fp:
            for token, target in iter_sentence(fp=fp, config=cls.Config, sep=' '):
                yield dict(token=token, target=convert_scheme(list(target)))

    @classmethod
    def load(cls, plm: PLM, **kwargs):