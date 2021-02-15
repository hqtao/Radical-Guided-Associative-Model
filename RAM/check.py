# coding: utf-8
# 2020/1/13 @ tongshiwei
import mxnet as mx
from CangJie.SE.bert import CharBert
from longling import AsyncLoopIter
from longling import print_time, load_jsonl
from tqdm import tqdm
import time

# cb = CharBert(mx.gpu(3))
# cb2 = CharBert(mx.gpu(2))


# @AsyncLoopIter.wrap
def iterator():
    src = "/home/tongshiwei/RAM/data/ctc32/test_bert_bs_s.json"
    for i, line in enumerate(load_jsonl(src)):
        if i > 1000:
            break
        yield line


if __name__ == '__main__':
    data = iterator()
    with print_time("testing"):
        for _ in tqdm(data):
            time.sleep(0.001)
