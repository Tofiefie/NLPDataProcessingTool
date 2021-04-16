import itertools
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from datasets import DownloadConfig
from datasets import DownloadManager
from torch.utils import data

from torchglyph import data_dir
from torchglyph.sampler import SortishBatchSampler
from torchglyph.sampler import SortishDevSampler
from torchglyph.sampler import SortishSampler

logger = getLogger(__name__)


class DataStore(object, metaclass=ABCMeta):
    name: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError

    @classmethod
    def paths(cls, root: Path = data_dir, **kwargs) -> List[Path]:
        out = []

        dataset_name = getattr(cls, 'name', cls.__name__).lower()
        for url, *names in cls.urls(**kwargs):
            Download_manager = DownloadManager(
                dataset_name=dataset_name,
                download_config=DownloadConfig(
                    cache_dir=root / dataset_name,
                    download_desc=f'Downloading {url}',
                ),
            )

            archive = Path(Download_manager.download_and_extract(url))
            out.append(archive)

        return out

    @classmethod
    def load(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_tokenize_fn(cls, **kwargs):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            raise NotImplementedError

        return tokenize

    @classmethod
    def get_collate_fn(cls, **kwargs):
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            raise NotImplementedError

        return collate_fn

    @classmethod
    def predicate(cls, example) -> bool:
        raise NotImplementedError

    @classmethod
    def new(cls, **