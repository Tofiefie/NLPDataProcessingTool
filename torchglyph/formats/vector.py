from typing import IO
from typing import Tuple

import torch
from tqdm import tqdm


def loads_meta(string: str, *, sep: str = ' ') -> Tuple[int, int]:
    num_embeddings, embedding_dim = string.strip().split(sep=sep)
    return int(num_embeddings), int(embedding_dim)


