#!/usr/bin/env python
#encoding: utf-8 

import sys
import numpy as np

def load(fname):
    word2vec_dict = {}

    idx = 0
    dim = 0
    for line in open(fname,'r'):
        idx += 1
        sep = line.strip().split(' ',1)
        if len(sep) < 2:
            continue

        key = sep[0]
        feature = sep[1]

        if idx == 1 :
            dim = int(feature)

        feature_list = feature.split(' ')

        float_feature_list = [float(i) for i in feature_list]
        float_feature_list = np.array(float_feature_list, dtype=np.float32)

        word2vec_dict[key] = float_feature_list

    return word2vec_dict, dim 

if __name__=='__main__':
    w2v, dim = load(sys.argv[1])

    word = '<PAD/>'
    word = 'digit'
    if word in w2v:
        print w2v[word]


