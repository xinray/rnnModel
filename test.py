import tensorflow as tf
import numpy as np
import word2vec
import data_helpers

input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
train_ids = tf.placeholder(dtype=tf.int32, shape=[None])

#embedding = tf.Variable(np.identity(5, dtype=np.int32))
t1 = [[0.001,0.001,0.001,0.001,0.001],[0.002,0.002,0.002,0.002,0.0002]]
t2 = [[0.003,0.003,0.003,0.003,0.003],[0.004,0.004,0.004,0.004,0.0004]]
t3 = [[0.005,0.005,0.005,0.005,0.005],[0.006,0.006,0.006,0.006,0.0006]]

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




train_data, valid_data, vocabulary, vocabulary_inv = data_helpers.load_data("data/labeledMy.tsv.train", 56, 128, 0.9, None)

x, y, mask_x = train_data

return_x = x[0:1, :]

print(return_x[0])

embedding, embeddingDim = load_word2vec(vocabulary)

#embedding =  tf.concat([t1,t2,t3],0)
#input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
embedding = embedding.astype(np.float32)
print(embedding.dtype)

lstm_cell = tf.contrib.rnn.GRUCell(56)

cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*2)

#embedding layer
with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):

    #embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, train_ids)
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

# Recurrent network.
# output.shape = [batch_size, max_len, hidden_size]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#print(embedding.eval())
#print(sess.run(input_embedding, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))
print(sess.run(inputs, feed_dict={train_ids:return_x[0]}))

#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#print_tensors_in_checkpoint_file("save/model.ckpt-277290.index", None, True)

#tt1 = tf.placeholder(dtype=tf.float32,shape=[None])
#tt2 = tf.placeholder(dtype=tf.float32,shape=[None])
#tt = tf.concat([tt1,tt2], axis =1)
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
ttt = tf.concat([t1, t2], 0)
print("tt")
#print(sess.run(tt,feed_dict={tt1:t1,tt2:t2}))
print(sess.run(ttt))






