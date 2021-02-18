import os
import platform
import socket
from pathlib import Path

from matplotlib import pyplot as plt

data_dir = (Path.home() / '.torchglyph').resolve()
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

host_name: str = socket.gethost