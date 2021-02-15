# coding: utf-8
# 2020/1/10 @ tongshiwei
import pandas as pd
import pathlib

from longling import path_append, as_list
from CangJie import token2radical
from tqdm import tqdm

META = path_append(pathlib.PurePath(__file__).parents[1], "meta_data")
DEFAULT_CAW = [None]


def load_ctype(filename):
    if pathlib.Path(filename).suffix in {".xls", ".xlsx"}:
        _ret = {}
        for i, row in tqdm(pd.read_excel(filename).iterrows(), "reading from %s" % filename):
            _ret[row["字"]] = row["类别"]
        return _ret
    else:
        raise TypeError("cannot handle %s" % filename)


def load_raw(filename):
    if pathlib.Path(filename).suffix in {".xls", ".xlsx"}:
        _ret = {}
        for i, row in tqdm(pd.read_excel(filename).iterrows(), "reading from %s" % filename):
            aws = row["相关概念"]
            aws = [] if aws == "无" else aws.split("、")
            _ret[row["部首"]] = aws
        return _ret
    else:
        raise TypeError("cannot handle %s" % filename)


class CAW(object):
    def __init__(self, ctype_file, raw_file):
        self._cdict = load_ctype(ctype_file)
        self._raw = load_raw(raw_file)

    def __getitem__(self, item: (str, list)):
        if isinstance(item, str):
            if len(item) > 1:
                return self[[e for e in item]]
            else:
                if item not in self._cdict or self._cdict[item] in {"汉字部件", "指事字", "象形字"}:
                    return None
                else:
                    radical = token2radical(item)
                    if radical in self._raw:
                        return self._raw[radical]
                    else:
                        return None
        elif isinstance(item, list):
            _ret = []
            for e in item:
                aw = self[e]
                if aw:
                    if isinstance(aw, list):
                        _ret.extend(aw)
                    else:
                        _ret.append(aw)
            return _ret
        else:
            raise TypeError("cannot handle %s" % type(item))

    def __call__(self, char: (str, list)):
        aws = self[char]
        return as_list(list(set(aws))) if aws else []


def caw(char):
    if DEFAULT_CAW[0] is None:
        DEFAULT_CAW[0] = CAW(
            path_append(META, "character-decomposition.xlsx"), path_append(META, "radical meaning.xlsx")
        )
    return DEFAULT_CAW[0](char)


if __name__ == '__main__':
    print(caw(["八月胡天即飞雪", "忽如一夜春风来"]))
