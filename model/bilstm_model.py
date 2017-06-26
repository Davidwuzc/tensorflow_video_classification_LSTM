from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

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
  """

  def __init__(self, is_training, input_, config):
    self._input       = input_
    self._config      = config
    self._is_training = is_training

    self._init_model()

  def _init_model(self):
    # Create multiple forward lstm cell
    cell_fw = rnn.MultiRNNCell(
      [rnn.BasicLSTMCell(self._config['hidden_size']) 
        for _ in range(self._config['num_layers'])]) 

    # Create multiple backward lstm cell
    cell_bw = rnn.MultiRNNCell(
      [rnn.BasicLSTMCell(self._config['hidden_size']) 
        for _ in range(self._config['num_layers'])]) 

    inputs = self._input.input_data

    # Add dropout layer to the input data 
    if self._is_training and self._config['keep_prob'] < 1:
      intpus = [tf.nn.dropout(single_input, self._config['keep_prob'])
                    for single_input in inputs]

    self._outputs, _, _ = rnn.static_bidirectional_rnn(
                            cell_fw, cell_bw, inputs, dtype=tf.float32)

    # Hidden layer weights => 2*hidden_size because of forward + backward cells
    softmax_w = tf.get_variable("softmax_w",
      [2*self._config['hidden_size'], self._config['num_classes']])
    softmax_b = tf.get_variable("softmax_b", [self._config['num_classes']])

    # Linear activation, using rnn inner loop last output
    #   logit shape: [batch_size, num_classes]
    self._logits = tf.matmul(self._outputs[-1], softmax_w) + softmax_b
 
    # Define loss
    # Required targets shape: [batch_size, num_classes] (one hot vector)
    self._cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, 
                                              labels=self._input.targets))
    # Evaluate model
    self._correct_pred = tf.equal(tf.argmax(self._logits, 1), 
                                  tf.argmax(self._input.targets, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    # Define optimizer
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