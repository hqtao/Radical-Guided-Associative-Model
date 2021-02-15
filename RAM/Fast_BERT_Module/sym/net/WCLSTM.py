# coding: utf-8
# 2020/1/2 @ tongshiwei
import mxnet as mx
from longling.ML.MxnetHelper.gallery.layer.attention import DotProductAttentionCell
from mxnet import gluon
from .net import EmbeddingLSTM


class WCLSTM(EmbeddingLSTM):
    def __init__(self, class_num, fc_dropout=0.0, **kwargs):
        r"""Baseline: 仅包含词和字，不包括部首的网络模型"""
        super(WCLSTM, self).__init__(**kwargs)
        self.word_length = None

        with self.name_scope():
            self.dropout = gluon.nn.Dropout(fc_dropout)
            self.fc = gluon.nn.Dense(class_num)

    def hybrid_forward(self, F, cls, *args, **kwargs):
        fc_in = self.dropout(cls)

        return self.fc(fc_in)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length
