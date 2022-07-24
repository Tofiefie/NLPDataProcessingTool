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
    return cache.parent / f'{cache.name}.{hf_hash(__torchglyph=DEBUG, **kwargs)}'


def cache_folder(path: Path, **kwargs) -> Path:
    cache = path.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    return cache / hf_hash(__torchglyph=DEBUG, **kwargs)


def all_exits(path: Path, *names: str) -> bool:
    for name in names:
        if not (path / name).exists():
            return False

    return True


def is_dataset_folder(path: Path) -> bool:
    path = path / DATASET_INFO_FILENAME
    return path.is_file() and path.exists()


def is_dataset_dict_folder(path: Path) -> bool:
    path = path / DATASETDICT_JSON_FILENAME
    return path.is_file() and path.exists()


@contextmanager
def lock_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.resolve() / '.lock')):
        yield


def load_json(path: Path) -> Any:
    with path.open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def load_args(out_dir: Path, name: str = ARGS_JSON) -> Any:
    return load_json(path=out_dir / name)


def load_sota(out_dir: Path, name: str = SOTA_JSON) -> Any:
    return load_json(path=out_dir / name)


def save_json(path: Path, **kwargs) -> None:
    data = {}
    if not path.exists():
        logger.info(f'saving to {path}')
    else:
        with path.open(mode='r', encoding='utf-8') as fp:
            data = json.load(fp=fp)

    with path.open(mode='w', encoding='utf-8') as fp:
        json.dump({**data, **kwargs}, fp=fp, indent=2, ensure_ascii=False)


def save_args(out_dir: Path, name: str = ARGS_JSON, **kwargs) -> None:
    return save_json(path=out_dir / name, **kwargs)


def save_sota(out_dir: Path, name: str = SOTA_JSON, **kwargs) -> None:
    return save_json(path=out_dir / name, **kwargs)


def load_checkpoint(name: str = CHECKPOINT_PT, strict: bool = True, *, out_dir: Path, **kwargs) -> None:
    state_dict = torch.load(out_dir / name, map_location=torch.device('cpu'))

    for name, module in kwargs.items():  # type: str, nn.Module
        logger.info(f'loading {name}.checkpoint from {