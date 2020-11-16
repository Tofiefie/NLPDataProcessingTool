import os
from pathlib import Path

from torchdevice import set_cuda_visible_devices

num_devices = int(os.environ.get('NUM_DEVICES', 0))
if num_devices > 0:
    set_cuda_visible_devic