# coding: utf-8
# create by tongshiwei on 2019/4/12

import numpy as np
from longling import loading, path_append
import mxnet as mx
from tqdm import tqdm
from gluonnlp.data import FixedBucketSampler, PadSequence
from CangJie.utils.embeddings import load_embedding, token_to_idx
from RAM.utils import caw
from longling import AsyncLoopIter, CacheAsyncLoopIter, iterwrap

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


@AsyncLoopIter.wrap
def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        from CangJie.utils.testing import pseudo_sentence

        sentences = pseudo_sentence(1000, 20)

        c = [[0] * 768 for s in sentences]

        labels = [random.randint(0, 32) for _ in sentences]

        return zip(c, labels)

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


@iterwrap(tank_size=1024)
def extract(data_src):
    for ds in loading(data_src):
        label = ds['label']

        cls = ds["b"]

        yield cls, label


@iterwrap()
def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size

    batch = []

    def format_batch(_batch):
        cls, label = list(zip(*_batch))

        return cls, label

    for data in raw_data:
        if len(batch) == batch_size:
            cls, label = format_batch(batch)
            batch = []
            yield cls, label
        else:
            batch.append(data)

    if batch:
        yield format_batch(batch)


@iterwrap("MemoryIter")
def load(transformed_data, params):
    for _data in transformed_data:
        _ret = []
        for i, feature in enumerate(_data):
            if i == 1:
                _ret.append(mx.nd.array(feature, dtype=np.int))
            else:
                _ret.append(mx.nd.array(feature))

        yield _ret


def etl(filename, params):
    raw_data = extract(filename)
    transformed_data = transform(raw_data, params)
    return load(transformed_data, params)


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    from longling import print_time
    from CangJie import PAD_TOKEN
    import os

    filename = "../../data/News_dataset/test_bert_bs.json"
    print(os.path.abspath(filename))

    parameters = AttrDict({"batch_size": 128, "padding": PAD_TOKEN}, num_buckets=100, fixed_length=None)
    for data in tqdm(etl(filename, params=parameters)):
        pass
