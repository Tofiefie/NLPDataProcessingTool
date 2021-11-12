
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