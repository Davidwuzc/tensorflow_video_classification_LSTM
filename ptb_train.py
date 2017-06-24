"""A binary to train BiLSTM on the PTB data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from data import ptb_data
from model.bilstm_model import BiLSTM

tf.app.flags.DEFINE_string("data_path", None,
          "Where the training/test data is stored.")
tf.app.flags.DEFINE_string("save_path", None,
          "Model output directory.")
FLAGS = tf.app.flags.FLAGS


class Config(object):
  """Configuration"""
  init_scale = 0.1
  learning_rate = 0.5
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 200
  max_epoch = 2
  max_max_epoch = 6
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 20
  num_classes = 10000


class PTBInput(object):
  """The input data."""
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    # input_data size: [batch_size, num_steps]
    # targets size: [batch_size, num_steps]
    self.input_data, self.targets = ptb_data.ptb_producer(
      data, batch_size, num_steps, name=name)

    # Data preprocessing: input_data
    #   [batch_size, num_steps] => num_steps * [batch_size, hidden_size]
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding",
        [config.num_classes, config.hidden_size],
        dtype=tf.float32)
    self.input_data = tf.nn.embedding_lookup(embedding, self.input_data)
    self.input_data = [tf.squeeze(input_step, [1])
      for input_step in tf.split(1, num_steps, self.input_data)]

    # Data preprocessing(one hot convertion): targets
    #   [batch_size, num_steps] => [batch_size * num_steps]
    self.targets = tf.reshape(self.targets, [-1])

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""

  start_time = time.time()
  costs = 0.0
  iters = 0

  fetches = {
    "cost": model.cost,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    vals = session.run(fetches)
    cost = vals["cost"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f -- perplexity: %.3f -- speed: %.0f wps" %
        (step*1.0 / model.input.epoch_size,
        np.exp(costs / iters),
        iters*model.input.batch_size / (time.time()-start_time)))

  return np.exp(costs / iters)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  # Tranfer the PTB raw data to corresponding number
  raw_data = ptb_data.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = Config()
  eval_config = Config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                          config.init_scale)

    with tf.name_scope('Train'):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        model = BiLSTM(True, train_input, config)
      tf.scalar_summary("Training Loss", model.cost)
      tf.scalar_summary("Learning Rate", model.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        mvalid = BiLSTM(False, valid_input, config)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        mtest = BiLSTM(False, test_input, config)

    sv = tf.train.Supervisor(logdir = FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        model.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
        train_perplexity = run_epoch(session, model, eval_op=model.train_op,
                       verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
  tf.app.run()
