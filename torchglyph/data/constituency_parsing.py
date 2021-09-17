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
                'word': encode_batch(examples['word'], tokenizer=word_tokenizer),
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
                'word': cat_sequence([example['word'] for example in examples]).to(device=device),
                'token': cat_sequence([example['token'] for example in examples]).to(device=device),
                'segment_size': cat_sequence([example['segment_size'] for example in examples]).to(device=device),
                'start': cat_sequence([example['start'] for example in examples]).to(device=device),
                'end': cat_sequence([example['end'] for example in examples]).to(device=device),
                'target': cat_sequence([example['target'] for example in examples]).to(device=device),
            }

        return collate_fn

    @classmethod
    def load_split(cls, path: Path, **kwargs):
        for word, tree in iter_ptb(path=path, do_binarize=True, do_factorize=True):
            start, end, target = zip(*tree)
            yield dict(word=word, start=start, end=end, target=target)

    @classmethod
    def load(cls, plm: PLM, **kwargs):
        cache = cache_folder(path=data_dir / cls.name, plm=plm.pretrained_model_name)
        word_cache = str(cache / WORD_FILENAME)
        target_cache = str(cache / TARGET_FILENAME)

        with lock_folder(path=cache):
            if not all_exits(cache, DATASETDICT_JSON_FILENAME, WORD_FILENAME, TARGET_FILENAME):
                train, validation, test = cls.paths(**kwargs)
                ds = DatasetDict(
                    train=Dataset.from_list(list(cls.load_split(path=train, **kwargs))),
                    validation=Dataset.from_list(list(cls.load_split(path=validation, **kwargs))),
                    test=Dataset.from_list(list(cls.load_split(path=test, **kwargs))),
                )

                word_tokenizer = train_word_tokenizer(
                    get_iterator(ds['train'], ds['validation'], column_names=['word']),
                    pre_tokenizer=False, unk_token='<unk>',
                )

                target_tokenizer = train_word_tokenizer(
                    get_iterator(ds['train'], ds['validation'], column_names=['target']),
                    pre_tokenizer=False, unk_token='S',
                )

                tokenize_fn = cls.get_tokenize_fn(
                    plm=plm,
                    word_tokenizer=word_tokenizer,
                    target_tokenizer=t