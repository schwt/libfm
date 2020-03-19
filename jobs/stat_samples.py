#!/usr/bin/env python
# coding:utf-8
from collections import defaultdict
"""
统计特征值分布
"""

f_train = "./data/train_sample.txt"
f_stat = "./stat.txt"

def load_data(f):
    buff = []
    with open(f) as rf:
        for line in rf.readlines():
            feat_val = {int(k): float(v) for k, v in (x.split(":") for x in line.strip().split()[1:])}
            buff.append(feat_val)
    print("#samples:", len(buff))
    return buff

def stat_accum(data):
    feat_vals = defaultdict(list)
    for feat_val in data:
        for fid, val in feat_val.items():
            feat_vals[fid].append(val)
    print("#fid:", len(feat_vals))
    return feat_vals

def stat_feat_val(feat_val):
    lists = sorted(feat_val.items(), key = lambda x: x[0])
    with open(f_stat, 'w') as wf:
        for feat, vals in lists:
            mean = sum(vals)/len(vals)
            wf.write("fid=%d,\tn=%d,\tavg=%g,\t(%g,\t%g)\n" % (feat, len(vals), mean, min(vals), max(vals)))
    print("done.")

data = load_data(f_train)
feat_val = stat_accum(data)
stat_feat_val(feat_val)

