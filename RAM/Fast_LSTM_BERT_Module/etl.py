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
        from CangJie import tokenize

        sentences = pseudo_sentence(1000, 20)

        def feature2num(token):
            if isinstance(token, list):
                return [feature2num(_token) for _token in token]
            else:
                return random.randint(0, 10)

        def feature2array(token):
            if isinstance(token, list):
                return [feature2array(_token) for _token in token]
            else:
                return [0] * _cfg.bert_dim

        w = [feature2array([c for c in s]) for s in sentences]
        c = [[0] * _cfg.bert_dim for s in sentences]
        aw = [feature2num(caw(s)) for s in sentences]

        labels = [random.randint(0, 32) for _ in sentences]

        return zip(w, c, aw, labels)

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


@AsyncLoopIter.wrap
def extract(data_src, embeddings):
    for ds in loading(data_src):
        label = ds['label']

        # seq = ds["s"][0]
        seq = ds["s"]
        cls = ds["b"]
        aw = token_to_idx(embeddings["w"], caw("".join(ds["c"])))

        yield seq, cls, aw, label


@AsyncLoopIter.wrap
def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size
    fixed_length = params.fixed_length

    batch = []

    def format_batch(_batch):
        seq, cls, aw, label = list(zip(*_batch))

        def padding(feature, pad_val):
            max_len = max(
                [len(fea) for fea in feature]
            ) if not fixed_length else fixed_length
            padder = PadSequence(max_len, pad_val=pad_val)
            feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
            return feature, mask

        seq, seq_mask = padding(seq, [0] * params.bert_dim)
        aw, aw_mask = padding(aw, 0)
        return seq, cls, aw, seq_mask, aw_mask, label

    for data in raw_data:
        if len(batch) >= batch_size:
            seq, cls, aw, seq_mask, aw_mask, label = format_batch(batch)
            batch = []
            yield seq, cls, aw, seq_mask, aw_mask, label
        batch.append(data)

    if batch:
        yield format_batch(batch)


@iterwrap("MemoryIter")
def load(transformed_data, params):
    for _data in transformed_data:
        _ret = []
        for i, feature in enumerate(_data):
            if i == 5:
                _ret.append(mx.nd.array(feature, dtype=np.int))
            else:
                try:
                    _ret.append(mx.nd.array(feature))
                except (TypeError, ValueError):
                    for i, fea in enumerate(feature):
                        for j, f in enumerate(fea):
                            assert len(f) == params.bert_dim, "%s %s %s" % (i, j, len(f))
                    exit(0)

        yield _ret


def etl(filename, embeddings, params):
    raw_data = extract(filename, embeddings)
    transformed_data = transform(raw_data, params)
    return load(transformed_data, params)


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    from longling import print_time
    from CangJie import PAD_TOKEN
    import os

    filename = "../../data/Fudan/data/test_bert_bs_torch_roberta_large.json"
    print(os.path.abspath(filename))

    vec_root = "../../data/vec/"
    with print_time("loading embedding"):
        _embeddings = load_embedding(
            {
                "w": path_append(vec_root, "word.vec.dat"),
            }
        )

    # for data in tqdm(extract(filename, _embeddings)):
    #     pass

    parameters = AttrDict({"batch_size": 32, "padding": PAD_TOKEN, "bert_dim": 1024}, num_buckets=100, fixed_length=None)
    count = 0
    for data in tqdm(etl(filename, _embeddings, params=parameters)):
        count += len(data[0])

    print(count)