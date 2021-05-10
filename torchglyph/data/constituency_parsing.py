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
from torc