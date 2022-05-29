import gzip
import json
import logging
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple

import torch
from datasets.config import DATASETDICT_JSON_FILENAME
from datasets.config import DATASET_INFO_FILENAME
from datasets.download import DownloadConfig
from datasets.download import DownloadManager
from datasets.fingerprint import Hasher
from filelock import FileLock
from torch import nn

from torchglyph import DEBUG
from torchglyph import data_dir

logger = logging.getLogger(__name__)

ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'
CHECKPOINT_PT = 'checkpoint.pt'


def download_and_extract(url: str, name: str, root: Path = data_dir) -> Path:
    manager = DownloadManager(
        dataset_name=name,
        download_config=DownloadConfig(
            cache_dir=root / name,
            download_desc=f'Downloading {url}',
        ),
    )

    return Path(manager.download_and_extract(url))


def hf_hash(**kwargs) -> str:
    hasher = Hasher()

    for key, value in sorted(kwargs.items()):
        hasher.update(key)
        hasher.update(value)

    return hasher.hexdigest()


def cache_file(path: Path, **kwargs) -> Path:
    cache = path.resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    return cache.paren