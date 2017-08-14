import sys
import numpy as np
import tensorflow as tf

from six.moves import cPickle
import pickle

import argparse
import time
import os

from rnn_model3 import RNN_Model
import data_helpers
import word2vec

def predict_epoch(model, session, data):
    x, y, mask_x = data

    feed_dict={}
    feed_dict[model.input_data] = x
    feed_dict[model.mask_x] = mask_x

    model.assign_new_batch_size(session, len(x))

    fetches = [model.probs]

    state = session.run(model._initial_state)

    #for i , (c,h) in enumerate(model._initial_state):
    #    feed_dict[c]=state[i].c
    #    feed_dict[h]=state[i].h

    probs = session.run(fetches, feed_dict)

    return np.array(probs[0])

def load_word2vec(vocabulary) :
    w2vModel, embeddingDim = word2vec.load('./data/dingdong.w2v.txt')

    vocabSize = len(vocabulary)+2

    embeddingUnknown = [0 for i in range(embeddingDim)]

    embeddingWeights = np.zeros((vocabSize, embeddingDim))

    for word, index in vocabulary.items():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            e = embeddingUnknown

        embeddingWeights[index, :] = e

    return embeddingWeights, embeddingDim


def main():

    with open(os.path.join("save_w2v", 'config.pkl'), 'rb') as f:
        args = cPickle.load(f)
    args.keep_prob = 1
    args.learning_phase = 0
    print args
    
    dictfile = args.train_path +".dict"
    pdictfile = open(dictfile,'r')
    voc = pickle.load(pdictfile)
    pdictfile.close()

    test_data, _,vocabulary, vocabulary_inv  = data_helpers.load_data(args.test_path, args.num_step, args.batch_size, 1.0, voc)
    #x, y, _ = test_data
    #print "predict x:", len(x)

    embeddingWeights, embeddingDim = load_word2vec(vocabulary)
    model = RNN_Model(args,embeddingWeights,  is_training=False)

    '''
    for k in x:
        li = k.tolist()
        p_li = [str(x) for x in li]
        print " ".join(p_li)

    print
    print
    print
    '''

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            probs = predict_epoch(model, sess, test_data)
            idx = probs.argmax(axis=-1)    
            print "labels\tscore"
            for i in range(len(probs)):
                label = idx[i]
                print "%s\t%s" % (idx[i], probs[i][label])  

if __name__ == '__main__':
    main()
