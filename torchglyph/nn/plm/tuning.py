
from functools import singledispatch

from torch import nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.bart.modeling_bart import BartDecoderLayer
from transformers.models.bart.modeling_bart import BartEncoderLayer
from transformers.models.bart.modeling_bart import BartModel
from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertOutput
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.bert.modeling_bert import BertSelfOutput
from transformers.models.mbart.modeling_mbart import MBartAttention
from transformers.models.mbart.modeling_mbart import MBartDecoderLayer
from transformers.models.mbart.modeling_mbart import MBartEncoderLayer
from transformers.models.mbart.modeling_mbart import MBartModel
from transformers.models.roberta.modeling_roberta import RobertaIntermediate
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaOutput
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfOutput


def full(*, self: nn.Module, **kwargs):
    self.requires_grad_(True)


def qof(*, self: nn.Module, **kwargs):
    self.requires_grad_(False)
    qof_recur(self, **kwargs)


@singledispatch
def qof_recur(self: nn.Module, **kwargs):
    pass


@qof_recur.register
def qof_bert_model(self: BertModel, **kwargs):
    for module in self.modules():
        if self is not module:
            qof_recur(module, **kwargs)


@qof_recur.register
def qof_bert_self_attention(self: BertSelfAttention, **kwargs):
    self.query.requires_grad_(True)


@qof_recur.register
def qof_bert_self_output(self: BertSelfOutput, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_bert_intermediate(self: BertIntermediate, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_bert_output(self: BertOutput, **kwargs):