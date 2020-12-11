from pathlib import Path

from setuptools import find_packages
from setuptools import setup

name = 'torchglyph'

root_dir = Path(__file__).parent.resolve()
with (root_dir / 'requirements.txt').open(mode='r', encoding='utf-8') as fp:
    install_requires = [install_require.strip() for install_require in fp]

setup(
    name=name,
    version