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

            a