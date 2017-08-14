import tensorflow as tf
import functools
import numpy as np
import math
from keras import backend as K
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def grouped_dropout(outputs,duty_ratios,is_learning):
    def output_func(idx):
        no = [tf.zeros_like(o) for o in outputs]
        no[idx] = outputs[idx] * len(outputs)
        return no

    # def false_func():
    #  return tf.zeros_like(tagger), tagger2*2
    duty_ratios = [math.log(dr) for dr in duty_ratios]
    tensor_group_idx = tf.multinomial([duty_ratios], 1)[0][0]
    dropout_outputs = tf.cond(tf.equal(is_learning, True),
                              lambda: tf.case(
                                  {tf.equal(tensor_group_idx, 0): lambda: output_func(0),
                                   tf.equal(tensor_group_idx, 1): lambda: output_func(1),
                                   tf.equal(tensor_group_idx, 2): lambda: output_func(2)},
                                  lambda: outputs),
                              lambda: outputs)
    return dropout_outputs


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
        self.reuse = None
        self.num_step = config.num_step
        self.input_data = tf.placeholder(tf.int32, [None, self.num_step])
        self.target = tf.placeholder(tf.int64, [None])
        self.mask_x = tf.placeholder(tf.int32, [None, self.num_step])
        #self.keep_prob = tf.placeholder(tf.float32)
        self.class_num = config.class_num
        self.hidden_neural_size = config.hidden_neural_size
        self.vocabulary_size = config.vocabulary_size
        self.embed_dim = config.embed_dim
        self.hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)
        self.regs = None
        self.inputs = None
        self.inputs_w2v = None

        self.setup_embedding()
        #self.setup_embedding_w2v(embedding_mat)

        self.output = self.lstm_model(self.inputs,self.length)
        #self.output_w2v = self.lstm_model(self.inputs_w2v,self.length)

        self.logits = self.setup_output(self.output)
        #self.logits = self.setup_output(self.output,self.output_w2v)

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

    # embedding layer
    def setup_embedding(self):
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            self.embedding = tf.get_variable("embedding", [self.vocabulary_size, self.embed_dim], dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            self.regs = tf.nn.l2_loss(self.embedding) * 0.001

    def setup_embedding_w2v(self,embedding_mat):
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
            embedding_mat = embedding_mat.astype(np.float32)
            self.embedding_w2v = tf.get_variable("w2v_embedding", initializer=embedding_mat)
            self.inputs_w2v = tf.nn.embedding_lookup(self.embedding_w2v, self.input_data)

    def lstm_model(self, inputs, length):
        with tf.variable_scope('lstmencode', reuse=(self.reuse)):
            # build LSTM network
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size)
            lstm_cell = tf.contrib.rnn.GRUCell(self.hidden_neural_size)
            if self.keep_prob < 1:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell, output_keep_prob=self.keep_prob
                    # lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob
                )

            # cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*hidden_layer_num, state_is_tuple=True)
            #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.hidden_layer_num)

            #self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

	    self._initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            if self.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            # Recurrent network.
            # output.shape = [batch_size, max_len, hidden_size]
            output, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                inputs,
                dtype=tf.float32,
                sequence_length=length
            )
            last = self._last_relevant(output, length)
            return last

    def setup_output(self, last1):
        logits = _linear(last1, self.class_num, True, 0.0)
        return logits


"""
    def setup_output(self, last1, last2):
        last1, last2 = grouped_dropout([last1, last2], [0.5, 0.5], K.learning_phase())
        tagger = tf.concat([last1, last2], axis=2)
        with tf.variable_scope('output', reuse=(self.reuse)):
            logits = _linear(tagger, self.class_num, True, 0.0)
        return logits
"""





