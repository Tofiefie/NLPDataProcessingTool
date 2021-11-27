
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Type
from typing import Union

from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk
from torch.types import Device
from torchrua import cat_sequence
from transformers import PreTrainedTokenizer

from torchglyph import DEBUG
from torchglyph import data_dir
from torchglyph.data.abc import DataLoader
from torchglyph.data.abc import DataStore
from torchglyph.dist import get_device
from torchglyph.io import cache_folder
from torchglyph.io import is_dataset_dict_folder
from torchglyph.io import lock_folder
from torchglyph.nn.plm import MBartLarge
from torchglyph.nn.plm import PLM
from torchglyph.nn.plm import RobertaBase

logger = getLogger(__name__)


class Wmt(DataStore):
    name: str
    subset: str
    src_lang: str
    tgt_lang: str

    ratio: float
    min_length: int
    max_length: int

    @classmethod
    def get_tokenize_fn(cls, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer):

        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            src = src_tokenizer([example[cls.src_lang] for example in examples['translation']])['input_ids']
            tgt = tgt_tokenizer([example[cls.tgt_lang] for example in examples['translation']])['input_ids']

            return {
                cls.src_lang: src, f'{cls.src_lang}.size': [len(example) for example in src],
                cls.tgt_lang: tgt, f'{cls.tgt_lang}.size': [len(example) for example in tgt],
            }

        return tokenize

    @classmethod
    def predicate(cls, example: Dict[str, Any]) -> bool:

        src = example[cls.src_lang]
        tgt = example[cls.tgt_lang]

        if not cls.min_length <= len(src) <= cls.max_length:
            return False

        if not cls.min_length <= len(tgt) <= cls.max_length:
            return False

        if cls.ratio * len(src) < len(tgt):
            return False

        if cls.ratio * len(tgt) < len(src):
            return False

        return True

    @classmethod
    def get_collate_fn(cls, device: Device):

        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            src = cat_sequence([example[cls.src_lang] for example in examples]).to(device=device)
            tgt = cat_sequence([example[cls.tgt_lang] for example in examples]).to(device=device)
            return {'src': src, 'tgt': tgt, 'batch_size': len(examples)}

        return collate_fn

    @classmethod
    def load(cls, src_plm: PLM, tgt_plm: PLM):
        cache = cache_folder(path=data_dir / cls.name / cls.subset, **{
            cls.src_lang: src_plm.pretrained_model_name,
            cls.tgt_lang: tgt_plm.pretrained_model_name,
        })

        with lock_folder(path=cache):
            if not is_dataset_dict_folder(path=cache):
                if not DEBUG:
                    ds: DatasetDict = load_dataset(cls.name, cls.subset)
                else:
                    ds = DatasetDict(
                        train=load_dataset(cls.name, cls.subset, split='train[:1024]'),
                        validation=load_dataset(cls.name, cls.subset, split='validation[:1024]'),
                        test=load_dataset(cls.name, cls.subset, split='test[:1024]'),
                    )

                tokenize_fn = cls.get_tokenize_fn(
                    src_tokenizer=src_plm.tokenizer,
                    tgt_tokenizer=tgt_plm.tokenizer,
                )

                ds = ds.map(tokenize_fn, batched=True, remove_columns='translation')
                ds['train'] = ds['train'].filter(cls.predicate)
                ds['validation'] = ds['validation'].filter(cls.predicate)

                ds.set_format('torch', columns=[cls.src_lang, cls.tgt_lang])

                logger.info(f'data => {ds}')
                logger.info(f'saving to {cache}')
                ds.save_to_disk(str(cache))
                return ds

        logger.info(f'loading from {cache}')
        return load_from_disk(str(cache), keep_in_memory=True)

    @classmethod
    def new(cls, batch_size: int = 4096,