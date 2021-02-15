# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["fit_f", "eval_f"]

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from tqdm import tqdm

from longling.ML.MxnetHelper.toolkit.ctx import split_and_load


def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    cls, label = _data

    output = _net(cls)

    bp_loss = None
    for name, func in loss_function.items():
        loss = func(output, label)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = nd.mean(loss).asscalar()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def eval_f(_net, test_data, ctx=mx.cpu()):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    ground_truth = []
    prediction = []

    def evaluation_function(y_true, y_pred):
        evaluation_result = {}
        precsion, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred
        )
        evaluation_result.update(
            {"precision_%d" % i: precsion[i] for i in range(len(precsion))}
        )
        evaluation_result.update(
            {"recall_%d" % i: recall[i] for i in range(len(recall))}
        )
        evaluation_result.update(
            {"f1_%d" % i: f1[i] for i in range(len(f1))}
        )
        evaluation_result.update(
            {"Accuracy": accuracy_score(y_true, y_pred)}
        )
        return evaluation_result

    # for batch_data in tqdm(test_data, "evaluating"):
    #     ctx_data = split_and_load(
    #         ctx, *batch_data,
    #         even_split=False
    #     )
    #     for (word, word_radical, char, char_radical, word_mask,
    #          char_mask, label) in ctx_data:
    #         output = _net(
    #             word, word_radical, char, char_radical, word_mask, char_mask
    #         )
    #         pred = mx.nd.argmax(output, axis=1)
    #         ground_truth.extend(label.asnumpy().tolist())
    #         prediction.extend(pred.asnumpy().tolist())

    for batch_data in tqdm(test_data, "evaluating"):
        for i in range(len(batch_data)):
            if hasattr(batch_data[i], "shape") and batch_data[i].shape[-1] > 0:
                batch_data[i] = batch_data[i].as_in_context(ctx)

        cls, label = batch_data
        output = _net(
            cls
        )
        pred = mx.nd.argmax(output, axis=1)
        ground_truth.extend(label.asnumpy().tolist())
        prediction.extend(pred.asnumpy().tolist())

    return evaluation_function(ground_truth, prediction)


def fit_f(net, batch_size, batch_data,
          trainer, bp_loss_f, loss_function, loss_monitor=None,
          ctx=mx.cpu()):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_size: int
            The size of each batch
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    bp_loss_f: dict with only one value and one key
        The function to compute the loss for the procession
        of back propagation
    loss_function: dict of function
        Some other measurement in addition to bp_loss_f
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.

    Returns
    -------

    """

    # ctx_data = split_and_load(
    #     ctx, *batch_data,
    #     even_split=False
    # )
    #
    # with autograd.record():
    #     for _data in ctx_data:
    #         bp_loss = _fit_f(
    #             net, _data, bp_loss_f, loss_function, loss_monitor
    #         )
    #         assert bp_loss is not None
    #         bp_loss.backward()
    # trainer.step(batch_size)

    for i in range(len(batch_data)):
        if hasattr(batch_data[i], "shape") and batch_data[i].shape[-1] > 0:
            batch_data[i] = batch_data[i].as_in_context(ctx)

    with autograd.record():
        bp_loss = _fit_f(net, batch_data, bp_loss_f, loss_function, loss_monitor)
        bp_loss.backward()
    trainer.step(batch_size)
