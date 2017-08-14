import tensorflow as tf
import functools
import numpy as np

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class RNN_Model(object):

    @lazy_property
    def length(self):
        length = tf.reduce_sum(self.mask_x, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])

        ##get last time steps output ,  the last time steps is not padding token 

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)

        return relevant

    def __init__(self, config, embedding_mat, is_training=True):

        self.keep_prob = config.keep_prob
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)

        num_step = config.num_step
        self.input_data = tf.placeholder(tf.int32, [None, num_step])
        self.target = tf.placeholder(tf.int64, [None])
        self.mask_x = tf.placeholder(tf.int32, [None, num_step])
        #self.keep_prob = tf.placeholder(tf.float32)
	self.regs = None

        class_num = config.class_num
        hidden_neural_size = config.hidden_neural_size
        vocabulary_size = config.vocabulary_size
        embed_dim = config.embed_dim
        hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

        #build LSTM network
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size)
        lstm_cell = tf.contrib.rnn.GRUCell(hidden_neural_size)
        if self.keep_prob<1:
            lstm_cell =  tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob
                #lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob
            )

        #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*hidden_layer_num, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*hidden_layer_num)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            #embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
	    embedding_mat = embedding_mat.astype(np.float32) 
            embedding = tf.get_variable("embedding",initializer=embedding_mat)
	    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
	    self.regs = tf.nn.l2_loss(embedding)*0.001
            print inputs.get_shape()

        if self.keep_prob<1:
            inputs = tf.nn.dropout(inputs,self.keep_prob)

        # Recurrent network.
        # output.shape = [batch_size, max_len, hidden_size]
        output, _ = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=self.length
        )
        last = self._last_relevant(output, self.length) 

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [hidden_neural_size, class_num], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [class_num], dtype=tf.float32)
            self.logits = tf.matmul(last, softmax_w) + softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits+1e-10)
            self.loss = self.loss + self.regs
	    self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits, 1)
            correct_prediction = tf.equal(self.prediction, self.target)
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        self.probs = tf.nn.softmax(self.logits)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)

        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr:lr_value})

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size:batch_size_value})
