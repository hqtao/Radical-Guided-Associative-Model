# coding: utf-8
# 2020/1/10 @ tongshiwei

import numpy as np
import json
from tqdm import tqdm
from longling import rf_open, wf_open, print_time
from longling import path_append
import mxnet as mx
from CangJie.SE.bert import CharBert

cb = CharBert(mx.gpu(1))


def preparing(src, tar, bert_model):
    with rf_open(src) as f, wf_open(tar) as wf:
        for line in tqdm(f, "preparing %s -> %s" % (src, tar)):
            data = json.loads(line)
            _data = dict()
            _data["label"] = data["label"]
            _data["w"] = data["w"]
            _data["c"] = data["c"]
            bert_embedding = bert_model("".join(data["c"]))
            cls_encoding = bert_embedding[1].asnumpy().squeeze()
            _data["b"] = np.around(cls_encoding, decimals=6).tolist()
            print(json.dumps(_data), file=wf)


def preparing2(src, tar, bert_model):
    datas = []
    with rf_open(src) as f, wf_open(tar) as wf:
        for line in tqdm(f, "reading from %s" % src):
            data = json.loads(line)

            datas.append({"c": data["c"], "label": data["label"]})

        with print_time("sorting"):
            datas.sort(key=lambda x: len(x["c"]))

        for data in tqdm(datas, "writing to %s" % tar):
            _data = dict()
            _data["label"] = data["label"]
            _data["c"] = data["c"]
            bert_embedding = bert_model(data["c"])
            _data["s"] = bert_embedding[0].asnumpy().squeeze().tolist()
            _data["b"] = bert_embedding[1].asnumpy().squeeze().tolist()
            print(json.dumps(_data), file=wf)



if __name__ == '__main__':
    # preparing2("../data/ctc32/train.json", "../data/ctc32/train_bert_bs.json", cb)
    # preparing2("../data/ctc32/test.json", "../data/ctc32/test_bert_bs.json", cb)
    # preparing("../data/ctc32/train.json", "../data/ctc32/train_bert.json", cb)
    # preparing("../data/ctc32/test.json", "../data/ctc32/test_bert.json", cb)
    # dataset = "Fudan"
    dataset = "News"
    dataset += "_dataset"

    data_dir = path_append("../data", dataset)
    # with concurrent_pool("p") as e:  # or concurrent_pool("t", ret=ret)
    #     e.submit(
    #         preparing2,
    #         path_append(data_dir, "train.json"),
    #         path_append(data_dir, "train_bert_bs.json"),
    #         cb
    #     )
    #     e.submit(
    #         preparing2,
    #         path_append(data_dir, "train.json"),
    #         path_append(data_dir, "train_bert_bs.json"),
    #         cb
    #     )
    #     e.submit(
    #         preparing,
    #         path_append(data_dir, "test.json"),
    #         path_append(data_dir, "test_bert.json"),
    #         cb
    #     )
    #     e.submit(
    #         preparing,
    #         path_append(data_dir, "test.json"),
    #         path_append(data_dir, "test_bert_.json"),
    #         cb
    #     )

    preparing2(
        path_append(data_dir, "train.json"),
        path_append(data_dir, "train_bert_bs.json"),
        cb
    )
    preparing2(
        path_append(data_dir, "test.json"),
        path_append(data_dir, "test_bert_bs.json"),
        cb
    )
    # preparing(
    #     path_append(data_dir, "train.json"),
    #     path_append(data_dir, "train_bert.json"),
    #     cb
    # )
    # preparing(
    #     path_append(data_dir, "test.json"),
    #     path_append(data_dir, "test_bert.json"),
    #     cb
    # )

