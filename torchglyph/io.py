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
from datasets.fingerprint