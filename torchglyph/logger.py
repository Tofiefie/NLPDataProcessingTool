import logging
import sys
from logging import getLogger
from pathlib import Path

import colorlog

from torchglyph import DEBUG

logger = getLogger(__name__)

LOG_TXT = 'log.txt'


def clear_root(*, level: int) -> None:
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
        handler.close()

    return logging.root.setLevel(level=level)


def add_stream_handler(*, level: int, fmt: str) -> None:
    stream_handler = logging.StreamHa