from pathlib import Path

from setuptools import find_packages
from setuptools import setup

name = 'torchglyph'

root_dir = Path(__file__).parent.resolve()
with (root_dir / 'requirements.txt').open(mode='r', encoding='utf-8') as fp:
  