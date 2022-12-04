from logging import getLogger
from typing import List
from typing import Union

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig

from torchglyph.nn.plm import utils

logger = getLogger(__name__)


class PLM(object):
    mapping = {}
    checkpoints = {}

    def __init__(self, *, lang: str, **kwargs) -> None:
        super(PLM, self).__init__()

        self.lang = lang
        self.pretrained_model_name = self.checkpoints[lang]

        self._config = None
        self._tokenizer = None
        self._model = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.pretrained_model_name})'

