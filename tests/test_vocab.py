from typing import List

from torchglyph.vocab import WordPieceVocab
from torchglyph.vocab import WordVocab


def test_word_vocab00():
    vocab = WordVocab[str, int](unk_token='<unk>')
    vocab.train_from_iterator([
        