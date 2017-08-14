import sys
import pickle
from six.moves import cPickle
import numpy as np
import tensorflow as tf

import argparse
import time
import os

from rnn_model3 import RNN_Model
import data_helpers
import word2vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/labeledMy.tsv.train',
                       help='data directory containing input.txt')

    parser.add_argument('--test_path', type=str, default='data/labeledMy.tsv.test',
                       help='data directory containing input.txt')

    parser.add_argument('--w2v', type=str, default='data/dingdong.w2v.txt',
                       help='word2vec model path')

    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')

    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')

    parser.add_argument('--vocabulary_size', type=int, default=20000,
                       help='vocabulary_size')

    parser.add_argument('--embed_dim', type=int, default=128,
                       help='emdedding_dim')

    parser.add_argument('--batch_size', type=int, default=128,
                       help='minibatch size')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate')

    parser.add_argument('--hidden_neural_size', type=int, default=100,
                       help='size of RNN hidden state')

    parser.add_argument('--hidden_layer_num', type=int, default=1,
                       help='number of layers in the RNN')

    parser.add_argument('--num_step', type=int, default=56,
                       help='RNN sequence length')

    parser.add_argument('--keep_prob', type=float, default=0.2,
                       help='keep_prob')

    parser.add_argument('--class_num', type=int, default=20,
                       help='number of class')

    parser.add_argument('--max_grad_norm', type=int, default=5,
                       help='max_grad_norm')

    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')

    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')

    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')

    parser.add_argument('--learning_phase',type=int,default= 1,
			help='learning_phase')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    train(args)

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

def train(args):

    ###train data, validation data
    ###vocabulary, vocabulary_in
    train_data, valid_data, vocabulary, vocabulary_inv = data_helpers.load_data(args.train_path, args.num_step, args.batch_size, 0.9, None)

    train_x, train_y, train_mask_x = train_data
    val_x, val_y, val_mask_x = valid_data

    ###save file
    dictfile = args.train_path +".dict"
    pdictfile = open(dictfile,'w')
    pickle.dump(vocabulary, pdictfile,0)
    pdictfile.close() 

    fp = open('data/dict', "w")
    for word, index in vocabulary.items():
        fp.write(word+"\t"+str(index)+"\n")
    fp.close()

    print "train size: %s" % (len(train_x))
    print "batch size: %s" % (args.batch_size)

    embeddingWeights, embeddingDim = load_word2vec(vocabulary)
    
    ### build model
    model = RNN_Model(args, embeddingWeights, is_training=True)

    merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(args.log_dir)

    best_accuracy = 0
    with tf.Session() as sess:

        #train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        for e in range(args.num_epochs):
            speed = 0
            sys.stdout.flush()
            print "epochs: %d" % (e)

            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))

            state = sess.run(model._initial_state)

            for b, (x, y, mask_x, num_batches_per_epoch) in enumerate(data_helpers.batch_iter(train_data, batch_size=args.batch_size)): 

                model.assign_new_batch_size(sess, len(x))

                start = time.time()

                feed = {model.input_data: x, 
                        model.target: y, 
                        model.mask_x: mask_x}
                
                #for i , (c,h) in enumerate(model._initial_state):
                #    feed[c]=state[i].c
                #    feed[h]=state[i].h

                fetches = [model.cost, model.accuracy, model.train_op]
                train_loss, accuracy, _ = sess.run(fetches, feed)

                #train_writer.add_summary(summary, e * args.batch_size + b)

                speed = time.time() - start

                if (e * args.batch_size + b) % 100 == 0:
                    sys.stdout.flush()
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, train_acc = {:.3f}, validation_acc = {:.3f}" \
                        .format(e * num_batches_per_epoch + b + 1,
                                args.num_epochs * num_batches_per_epoch,
                                e, train_loss, speed, accuracy, 
                                model.accuracy.eval({model.input_data:val_x, model.target:val_y, model.mask_x:val_mask_x})))

            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = e * num_batches_per_epoch)
            best_accuracy = accuracy
            
            sys.stdout.flush()
            print("model saved to {}".format(checkpoint_path))

        #train_writer.close()

if __name__ == '__main__':
    main()
