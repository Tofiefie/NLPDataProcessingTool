from typing import Any
from typing import IO
from typing import Iterable
from typing import NamedTuple
from typing import Tuple
from typing import Type
from typing import get_type_hints

from torchglyph.formats.primitive import dumps_type
from torchglyph.formats.primitive import loads_type

Token = Tuple[Any, ...]
Sentence = Tuple[Tuple[Any, ...], ...]


def loads_token(string: str, *, config: Type[NamedTuple], sep: str = '\t') -> Token:
 