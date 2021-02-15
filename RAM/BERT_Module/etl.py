# coding: utf-8
# create by tongshiwei on 2019/4/12

import numpy as np
import warnings
from longling import loading, path_append
import mxnet as mx
from tqdm import tqdm
from gluonnlp.data import FixedBucketSampler, PadSequence
from CangJie.utils.embeddings import load_embedding, token_to_idx
from RAM.utils import caw

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        from CangJie.utils.testing import pseudo_sentence
        from CangJie import tokenize, characterize, token2radical

        sentences = pseudo_sentence(1000, 20)

        def feature2num(token):
            if isinstance(token, list):
                return [feature2num(_token) for _token in token]
            else:
                return random.randint(0, 10)

        w = [feature2num(list(tokenize(s))) for s in sentences]
        aw = [feature2num(caw(s)) for s in sentences]

        labels = [random.randint(0, 32) for _ in sentences]
        features = [w, sentences, aw]

        return features, labels

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


def extract(data_src, embeddings):
    word_feature = []
    sentences = []
    associate_word_feature = []
    features = [word_feature, sentences, associate_word_feature]

    labels = []
    for ds in tqdm(loading(data_src), "loading data from %s" % data_src):
        label = ds['label']

        w = token_to_idx(embeddings["w"], ds["w"])
        aw = token_to_idx(embeddings["w"], caw("".join(ds["c"])))

        if len(w) < 1:
            continue

        word_feature.append(w)
        sentences.append("".join(ds["c"]))
        associate_word_feature.append(aw)
        labels.append(label)

    return features, labels


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size
    padding = params.padding
    num_buckets = params.num_buckets
    fixed_length = params.fixed_length

    features, labels = raw_data
    word_feature, sentences, associate_word_feature = features
    batch_idxes = FixedBucketSampler(
        [len(word_f) for word_f in word_feature],
        batch_size, num_buckets=num_buckets
    )
    batch = []
    for batch_idx in batch_idxes:
        batch_features = [[] for _ in range(len(features))]
        batch_labels = []
        for idx in batch_idx:
            for i, feature in enumerate(batch_features):
                batch_features[i].append(features[i][idx])
            batch_labels.append(labels[idx])
        batch_data = []
        word_mask = []
        associate_word_mask = []
        for i, feature in enumerate(batch_features):
            if i in {0, 2}:
                max_len = max(
                    [len(fea) for fea in feature]
                ) if not fixed_length else fixed_length
                padder = PadSequence(max_len, pad_val=padding)
                feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
                if i == 0:
                    word_mask = mask
                elif i == 2:
                    associate_word_mask = mask
                batch_data.append(mx.nd.array(feature))
            else:
                batch_data.append(feature)
        batch_data.append(mx.nd.array(word_mask))
        batch_data.append(mx.nd.array(associate_word_mask))
        batch_data.append(mx.nd.array(batch_labels, dtype=np.int))
        batch.append(batch_data)
    return batch[::-1]


def load(transformed_data, params):
    return transformed_data


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
