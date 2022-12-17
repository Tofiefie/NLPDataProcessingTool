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

    @property
    def config(self) -> PretrainedConfig:
        if self._config is None:
            self._config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
            )

        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                src_lang=self.mapping.get(self.lang, None),
                use_fast=True,
            )

        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = Au