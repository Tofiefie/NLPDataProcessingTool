
from logging import getLogger
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple
from typing import Type

import torch
from tabulate import tabulate

from torchglyph.io import ARGS_JSON
from torchglyph.io import SOTA_JSON
from torchglyph.io import load_args
from torchglyph.io import load_sota
from torchglyph.logger import LOG_TXT

logger = getLogger(__name__)

SUMMARY_IGNORES = ('study', 'device', 'seed', 'hostname', 'port', 'checkpoint')
SUMMARY_IGNORES = SUMMARY_IGNORES + tuple(f'co-{ignore}' for ignore in SUMMARY_IGNORES)

