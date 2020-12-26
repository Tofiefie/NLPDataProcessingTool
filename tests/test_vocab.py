from typing import List

from torchglyph.vocab import WordPieceVocab
from torchglyph.vocab import WordVocab


def test_word_vocab00():
    vocab = WordVocab[str, int](unk_token='<unk>')
    vocab.train_from_iterator([
        'label1',
        'label2',
    ])

    index1 = vocab.encode('label1')
    index2 = vocab.encode('label2')
    index3 = vocab.encode('label3')

    assert vocab.inv(index1) == 'label1'
    assert vocab.inv(index2) == 'label2'
    assert vocab.inv(index3) == '<unk>'

    assert vocab.decode(index1) == 'label1'
    assert vocab.decode(index2) == 'label2'
    assert vocab.decode(index3) == '<unk>'

    index = vocab.encode_batch(['label1', 'label2', 'label3'])
    assert vocab.inv_batch(index) == ['label1', 'label2', '<unk>']
    assert vocab.decode_batch(index) == ['label1', 'label2', '<unk>']


def test_word_vocab01():
    vocab = WordVocab[str, List[int]](unk_token='<unk>')
    vocab.train_from_iterator([
        'this is the first sentence, and it is great',
        'anothe