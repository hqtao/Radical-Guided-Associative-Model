# coding: utf-8
# 2020/1/2 @ tongshiwei

from mxnet import gluon


class EmbeddingLSTM(gluon.HybridBlock):
    def hybrid_forward(self, F, w, c, aw, word_mask, charater_mask, associate_word_mask, *args, **kwargs):
        raise NotImplementedError
