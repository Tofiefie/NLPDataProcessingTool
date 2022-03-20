from typing import IO
from typing import Tuple

import torch
from tqdm import tqdm


def loads_meta(string: str, *, sep: str = ' ') -> Tuple[int, int]:
    num_embeddings, embedding_dim = string.strip().split(sep=sep)
    return int(num_embeddings), int(embedding_dim)


def load_meta(fp: IO, *, sep: str = ' ') -> Tuple[int, int]:
    return loads_meta(fp.readline(), sep=sep)


def loads_vector(string: str, *, sep: str = ' '):
    token, *values = string.rstrip().split(sep=sep)
    return str(token), tuple(float(value) for value in values)


def load_vector(fp: IO, *, sep: str = ' '):
    return loads_vector(fp.readline(), sep=sep)


def load_word2vec(fp: IO, *, sep: str = ' '):
