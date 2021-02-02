import os
import platform
import socket
from pathlib import Path

from matplotlib import pyplot as plt

data_dir = (Path.home() / '.torchglyph').resolv