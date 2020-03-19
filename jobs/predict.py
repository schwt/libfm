#!/usr/bin/env python
# coding:utf-8
import numpy as np
from math import exp
"""
验证预测值
"""

f_model = "model.txt"
f_test  = "./data/spcz_test.txt"
f_out   = "predict.txt"

class FMModel():
    def __init__(self):
        self.w0 = 0
        self.dim = 0
        self.w = np.zeros(10)
        self.v = np.zeros((10,5))

    def load_model(self, f_model):
        flag = ''
        buff_w = []
        buff_v = []
        print("model loading...")
        with open(f_model) as rf:
            cnt = 0
            for line in rf.readlines():
                cnt += 1
                line = line.strip()
                if line == 'w0':
                    flag = 'w0'
                    continue
                if line == 'w':
                    flag = 'w'
                    continue
                if line == 'v':
                    flag = 'v'
                    continue
                if flag == 'w0':
                    self.w0 = float(line)
                    continue
                if flag == 'w':
                    buff_w.append(float(line))
                    continue
                if flag == 'v':
                    values = [float(s) for s in line.split("\t")]
                    buff_v.append(values)
                    continue
        self.w = np.array(buff_w)
        self.v = np.array(buff_v)
        self.dim = self.w.shape[0]
        print("model loaded.(#line=%s)" % cnt)
        print("w0 =", self.w0)
        print("#w =", self.w.shape)
        print("#v =", self.v.shape)

    def predict(self, features = {}):
        s = self.w0
        for fid, val in features.items():
            s += val * self.w[fid]
            for fid2, val2 in features.items():
                if fid2 == fid:
                    continue
                s += 0.5 * val * val2 * self.v[fid].dot(self.v[fid2])
        ret = 1 / (1 + exp(-s))
        print("pred:", s, ret)
        return ret

def predict_test(f_test, f_out, model):
    wf = open(f_out, 'w')
    with open(f_test) as rf:
        for line in rf.readlines():
            features = {int(k): float(v) for k, v in (x.split(":") for x in line.strip().split()[1:])}
            print("feat:", features)
            pred = model.predict(features)
            wf.write("%s\n" % pred)
            input()
    wf.close()
    print("predict done.")

model = FMModel()
model.load_model(f_model)
predict_test(f_test, f_out, model)


