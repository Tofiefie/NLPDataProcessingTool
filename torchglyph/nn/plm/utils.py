
from typing import List
from typing import Union

from transformers import PreTrainedTokenizer


def tokenize_sequence(text: str, *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                      add_prefix_space: bool = False, add_special_tokens: bool = True):
    return tokenizer(
        f' {text}' if add_prefix_space else text,
        add_special_tokens=add_special_tokens,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']


def tokenize_sequence_batch(text: List[str], *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                            add_prefix_space: bool = False, add_special_tokens: bool = True) -> List[List[int]]:
    return tokenizer(
        [f' {sequence}' for sequence in text] if add_prefix_space else text,
        add_special_tokens=add_special_tokens,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']


def add_prefix(sequence: List[str], add_prefix_space: bool):
    return [f' {token}' if index > 0 or add_prefix_space else token for index, token in enumerate(sequence)]