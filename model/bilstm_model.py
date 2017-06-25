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

  def __init__(self, is_training, input_, config, is_video=False):
    self._input  = input_
    self._config = config
    self._is_training = is_training

    self._init_model()

  def _init_model(self):
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._config.hidden_size,
                                                state_is_tuple=True)
    if self._is_training and self._config.keep_prob < 1:
      lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_fw_cell, output_keep_prob=self._config.keep_prob)
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_fw_cell]*self._config.num_layers, 
      state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._config.hidden_size,
                                                state_is_tuple=True) 
    if self._is_training and self._config.keep_prob < 1:
      lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_bw_cell, output_keep_prob=self._config.keep_prob)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_fw_cell]*self._config.num_layers, 
      state_is_tuple=True)

    inputs = self._input.input_data
    if self._is_training and self._config.keep_prob < 1:
      intpus = [tf.nn.dropout(single_input, self._config.keep_prob) 
                    for single_input in inputs]

    self._outputs, _, _ = tf.nn.bidirectional_rnn(
      cell_fw, cell_bw, inputs, dtype=tf.float32)

    softmax_w = tf.get_variable("softmax_w", 
      [2*self._config.hidden_size, self._config.num_classes])
    softmax_b = tf.get_variable("softmax_b", [self._config.num_classes])

    # Linear activation, using rnn inner loop last output
    #   logit shape: [batch_size, num_classes]
    self._logits = tf.matmul(self._outputs[-1], softmax_w) + softmax_b
    # Required targets shape: [batch_size, num_classes] (one hot vector)
    self._cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(self._logits, self._input.targets))
    self._correct_pred = tf.equal(tf.argmax(self._logits, 1), 
                                  tf.argmax(self._input.targets, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    self._lr = tf.Variable(0.0, trainable=False)
    self._train_op = tf.train.AdamOptimizer(
      learning_rate=self._lr).minimize(self._cost)

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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