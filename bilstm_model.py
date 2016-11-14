from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# ----------------------------------------------------------------------------
# LSTM Model
# ----------------------------------------------------------------------------
class BiLSTM(object):
    """Bidirectional LSTM neural network.

    Use this function to create the bidirection LSTM nerual network model

    Args:
        input_: DataInput class, 
            input_data: List, a list of tensor placeholder that represent 
                batches of input data required shape. 
                Size: num_steps * [batch_size, hidden_size]
            targets: Tensor, corresponding one hot vector groudtrue for 
                input, Size:[batch_size, num_classes] 
        is_training: Boolean, whether to apply a dropout layer or not
        config: configuration file
        is_video: Boolean, whether the input is a video or not, default is
            false
    """

    def __init__(self, is_training, input_, config, is_video = False):
        self._input = input_

        # Deifine the LSTM cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, 
                                                 forget_bias = 0.0, 
                                                 state_is_tuple = True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob = config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers,     
                                           state_is_tuple = True)

        inputs = self._input.input_data
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        self._outputs, state = tf.nn.rnn(cell, inputs, dtype = tf.float32)

        softmax_w = tf.get_variable("softmax_w", 
            [config.hidden_size, config.num_classes])
        softmax_b = tf.get_variable("softmax_b", [config.num_classes])

        if not is_video:
            # Tranformthe outputs
            #   num_steps * [batch_size, hidden_size] =>
            #   [batch_size * num_steps, hidden_size]
            self._outputs = tf.reshape(
                tf.concat(1, self._outputs), [-1, config.hidden_size])
        
        if is_video:
            # Linear activation, using rnn inner loop last output
            self._logits = tf.matmul(self._outputs[-1], softmax_w) + softmax_b
        else:
            self._logits = tf.matmul(self._outputs, softmax_w) + softmax_b

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self._logits],
            [self._input.targets],
            [tf.ones([config.batch_size * config.num_steps], dtype = tf.float32)])
        self._cost = tf.reduce_sum(loss) / config.batch_size

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable = False)
        self._train_op = tf.train.AdamOptimizer(
            learning_rate = self._lr).minimize(self._cost)

        self._new_lr = tf.placeholder(
            tf.float32, shape = [], name = "new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
    
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def cost(self):
        return self._cost        

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op