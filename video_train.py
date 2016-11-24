from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from video_input import DataInput
from bilstm_model import BiLSTM

FLAGS = tf.app.flags.FLAGS


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0

  fetches = {
    "cost": model.cost,
    "accuracy": model.accuracy
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    vals = session.run(fetches)
    cost = vals["cost"]
    accuracy = vals["accuracy"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % 10 == 9:
      print("accuracy: %.3f" % accuracy)
      print("%.3f -- perplexity: %.3f -- speed: %.0f vps" %
          (step * 1.0 / model.input.epoch_size,
           np.exp(costs / iters),
           iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def train(config, data):
  """video training procedure
  Args:
    config: the configuration class
    data: data class that implement the dataset.py interface
  """
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                          config.init_scale)
    with tf.name_scope('Train'):
      with tf.variable_scope('Model', reuse=None, initializer=initializer):
        train_input = DataInput(config=config, data=data)
        model = BiLSTM(True, train_input, config, is_video=True)
      tf.scalar_summary("Training Loss", model.cost)
      tf.scalar_summary("Learning Rate", model.lr)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        # Check if the one hot label is correct for corresponding image
        # a,b,c=session.run([train_input.filenames,train_input.labels,train_input.targets])
        # print("a:{} b:{} c:{}".format(a, b, c))

        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        model.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" %
            (i + 1, session.run(model.lr)))
        train_perplexity = run_epoch(session, model, eval_op=model.train_op,
                       verbose=True)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path,
                global_step=sv.global_step)