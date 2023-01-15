
from collections import defaultdict

from torchglyph.nn.plm import PLM


class BertBase(PLM):
    checkpoints = {
        'en': 'bert-base-cased',
        'de': 'bert-base-german-cased',
    }


class BertLarge(PLM):
    checkpoints = {
        'en': 'bert-large-cased',
    }


class MBertBase(PLM):