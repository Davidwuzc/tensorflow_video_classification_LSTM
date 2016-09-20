from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import image_processing
import settings

FLAGS = tf.app.flags.FLAGS

class SmallConfig(object):
  """Small config."""
  # Parameters
  learning_rate = 0.1
  training_iters = 100
  batch_size = FLAGS.batch_size
  display_step = 1
  row = FLAGS.image_size
  column = FLAGS.image_size
  channel = 3
  # Network parameters
  num_input = row * column * channel
  num_steps = FLAGS.sequence_size
  num_hidden = 200 # hidden layer number of features

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def BiLSTM(x, weights, biases):
  """Bidirectional LSTM neural network.

  Use this function to create the nerual network model

  Args:
    x: a tensor placeholder that represent batches of video
    weight: variable, all the weight variable of the model
    biases: variable, all the biases variable of the model

  Returns:
    pred: tensor. predition value calculated by the lastest model
  """
  # get the configuration
  config = get_config()

  # Prepare data shape to match `bidirectional_rnn` function requirements
  # Current data input shape: (batch_size, n_step, n_row, n_column, n_channel)
  # Required shape: 'num_steps' tensors list of shape (batch_size, num_input)
  # num_input is equal to n_row * n_column * n_channel
 
  # Reshape to (batch_size, n_step, num_input)
  x = tf.reshape(x, [config.batch_size, config.num_steps, config.num_input])
  # Permuting batch_size and n_step
  x = tf.transpose(x, [1, 0, 2])
  # Reshape to (n_step * batch_size, num_input)
  x = tf.reshape(x, [-1, config.num_input])
  # Split to get a list of 'num_steps' tensors of shape (batch_size, num_input)
  x = tf.split(0, config.num_steps, x)

  # Define lstm cells with tensorflow
  # Forward direction cell
  lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(config.num_hidden,
                                        forget_bias=1.0, state_is_tuple=True)
  # Backward direction cell
  lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(config.num_hidden,
                                        forget_bias=1.0, state_is_tuple=True)

  # Get lstm cell output
  try:
    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)
  except Exception: # Old TensorFlow version only returns outputs not states
    outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                    dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']

def train(dataset):
  # get the configuration settings
  config = get_config()
  num_classes = dataset.num_classes()

  # tf Graph inputs
  x = tf.placeholder("float", [
    None,
    config.num_steps,
    config.row,
    config.column,
    config.channel])
  y = tf.placeholder("float", [None, num_classes])

  # Define weights
  weights = {
    'out': tf.Variable(tf.random_normal([2*config.num_hidden, 
                                        num_classes]))
  }
  biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
  }

  pred = BiLSTM(x, weights, biases)

  # Define loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
  optimizer = tf.train.AdamOptimizer(
    learning_rate=config.learning_rate).minimize(cost)

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initializing the variables
  init = tf.initialize_all_variables()
  
  # coordinator for controlling queue threads
  coord = tf.train.Coordinator()

  # initialize the image and label operator
  images_op, labels_op, filenames_op = image_processing.distorted_inputs(
    dataset,
    batch_size=config.batch_size)

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)
    step = 1

    # start all the queue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Keep training until reach max iterations
    while step * config.batch_size < config.training_iters:

      # get the image and label data
      images, labels, filenames = sess.run([images_op, labels_op, filenames_op])

      # run the optimizer 
      sess.run(optimizer, feed_dict={x: images, y: labels})

      if step % config.display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: images, y: labels})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: images, y: labels})
        print("Iter " + str(step*config.batch_size) + ", Minibatch Loss= " + \
          "{:.6f}".format(loss) + ", Training Accuracy= " + \
          "{:.5f}".format(acc))
      step += 1

    # request to stop the input queue
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)

    print("Optimization Finished!")