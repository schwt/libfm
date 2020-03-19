#!/bin/sh
# -*- coding: utf8 -*-
dir=$(cd $(dirname $0); pwd)

bin=../bin/libFM
f_train=./data/spcz_sample.txt
f_train=./data/spcz_train.txt
f_test=./data/spcz_test.txt
iter=100
f_log=log.txt
f_out=out.txt
f_model=model.txt
eta=0.000000001
dim=8

${bin} -task c -method sgd -train ${f_train} -test ${f_test} -dim "1,1,${dim}" -regular '0,0,0.01' -learn_rate ${eta} -init_stdev 0.5 -iter ${iter} -rlog ${f_log} -out ${f_out} -save_model ${f_model}

# ${bin} -task c -method mcmc -train ${f_train} -test ${f_test} -dim '1,1,5' -iter ${iter} -rlog ${f_log} -out ${f_out}
