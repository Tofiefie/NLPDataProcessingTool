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
from torchg