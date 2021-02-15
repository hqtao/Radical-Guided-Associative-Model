# coding: utf-8
# create by tongshiwei on 2019/4/12

import numpy as np
from longling import loading, path_append, CacheAsyncLoopIter, AsyncLoopIter
import mxnet as mx
from tqdm import tqdm
from gluonnlp.data import FixedBucketSampler, PadSequence
from CangJie.utils.embeddings import load_embedding, token_to_idx
from CangJie.SE.bert import CharBert
from RAM.utils import caw

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


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

        c = [[_c for _c in s] for s in sentences]
        aw = [feature2num(caw(s)) for s in sentences]

        labels = [random.randint(0, 31) for _ in sentences]
        features = [c, aw]

        return features, labels

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


def extract(data_src, embeddings):
    sentence = []
    associate_word_feature = []
    features = [sentence, associate_word_feature]

    labels = []
    for ds in tqdm(loading(data_src), "loading data from %s" % data_src):
        label = ds['label']

        seq = ds["c"]
        aw = token_to_idx(embeddings["w"], caw(seq))

        sentence.append(seq)
        associate_word_feature.append(aw)
        labels.append(label)

    return features, labels


@AsyncLoopIter.wrap
def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data
    cb = CharBert(ctx=mx.gpu(1))

    batch_size = params.batch_size
    num_buckets = params.num_buckets
    fixed_length = params.fixed_length

    features, labels = raw_data
    sentence, associate_word_feature = features
    batch_idxes = FixedBucketSampler(
        [len(seq) for seq in sentence],
        batch_size, num_buckets=num_buckets
    )
    for batch_idx in batch_idxes:
        batch_features = [[] for _ in range(len(features))]
        batch_labels = []
        for idx in batch_idx:
            for i, feature in enumerate(batch_features):
                batch_features[i].append(features[i][idx])
            batch_labels.append(labels[idx])
        batch_data = []
        seq_mask = []
        associate_word_mask = []
        for i, feature in enumerate(batch_features):
            if i == 0:  # sentence to seq_encoding, cls_encoding
                max_len = max(
                    [len(fea) for fea in feature]
                ) if not fixed_length else fixed_length
                seq_encoding = []
                cls_encoding = []
                for seq in feature:
                    _seq_encoding, _cls_encoding = cb(seq)
                    seq_encoding.append(_seq_encoding.asnumpy().squeeze().tolist())
                    cls_encoding.append(_cls_encoding.asnumpy().squeeze().tolist())
                padder = PadSequence(max_len, pad_val=[0] * 768)
                seq_encoding, mask = zip(*[(padder(seq), len(seq)) for seq in seq_encoding])
                seq_mask = mask
                batch_data.append(seq_encoding)
                batch_data.append(cls_encoding)
            elif i == 1:  # aw: token to idx
                max_len = max(
                    [len(fea) for fea in feature]
                ) if not fixed_length else fixed_length
                padder = PadSequence(max_len, pad_val=0)
                feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
                associate_word_mask = mask
                batch_data.append(feature)
        batch_data.append(seq_mask)
        batch_data.append(associate_word_mask)
        batch_data.append(batch_labels)
        yield batch_data


@AsyncLoopIter.wrap
def load(transformed_data, params):
    for _data in transformed_data:
        _ret = []
        for i, feature in enumerate(_data):
            if i == 4:
                _ret.append(mx.nd.array(feature, dtype=np.int))
            else:
                _ret.append(mx.nd.array(feature))
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

    filename = "../../data/ctc32/train.json"
    print(os.path.abspath(filename))

    vec_root = "../../data/vec/"
    with print_time("loading embedding"):
        _embeddings = load_embedding(
            {
                "w": path_append(vec_root, "word.vec.dat"),
            }
        )

    for data in tqdm(extract(filename, _embeddings)):
        pass

    parameters = AttrDict({"batch_size": 128, "padding": PAD_TOKEN}, num_buckets=100, fixed_length=None)
    for data in tqdm(etl(filename, _embeddings, params=parameters)):
        pass
