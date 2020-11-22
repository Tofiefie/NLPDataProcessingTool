import os
from pathlib import Path

from torchdevice import set_cuda_visible_devices

num_devices = int(os.environ.get('NUM_DEVICES', 0))
if num_devices > 0:
    set_cuda_visible_devices(n=num_devices)

app_dir = Path(__file__).resolve().parent
project_dir = app_dir.parent
app_name = app_dir.name

system_data_dir = Path.home() / 'data'
system_data_dir.mkdir(parents=True, exist_ok=True)

project_out_dir = project_dir / 'out'
project_out_dir.mkdir(parents=True, exist_ok=True)

project_data_dir = project_dir / 'data'
project_data_dir.mkdir(parents=True, exist_ok=True)
