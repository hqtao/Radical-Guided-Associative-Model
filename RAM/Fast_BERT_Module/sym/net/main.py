# coding: utf-8
# 2020/1/3 @ tongshiwei


__all__ = ["get_net", "get_bp_loss"]

import mxnet as mx
from mxnet import gluon
from .WCLSTM import WCLSTM


def get_net(model_type, class_num, **kwargs):
    if model_type == "wclstm":
        return WCLSTM(class_num=class_num, **kwargs)
    else:
        raise TypeError("unknown model_type: %s" % model_type)


def get_bp_loss(**kwargs):
    return {"cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss()}
