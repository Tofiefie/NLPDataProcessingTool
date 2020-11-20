import os
from pathlib import Path

from torchdevice import set_cuda_visible_devices

num_devices = int(os.environ.get('NUM_DEVICES', 0))
if num_devices > 0:
    set_cuda_visible_devices(n=num_devices)

app_dir = Path(__file__).resolve().parent
project_dir = app_dir.parent
app_name = app_