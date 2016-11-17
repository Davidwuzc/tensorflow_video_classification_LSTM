"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from bilstm_model import BiLSTM
from kth_data import KTHData
import video_processing as vp

tf.app.flags.DEFINE_string("data_dir", None,
          "Where the training/validation data is stored.")
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


class KTHInput(object):
  """The input data."""
  def __init__(self, config, data):
    self.batch_size = batch_size = config.batch_size
    videos_op, labels_op, _ = vp.distorted_inputs(data, batch_size)

def main(_):
  if not FLAGS.data_dir:
    raise ValueError("Must set --data_dir to KTH data directory")

  train_data = KTHData('train')
  assert train_data.data_files()
  
  config = Config()
  config.num_classes = train_data.num_classes()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                          config.init_scale)
    with tf.name_scope('Train'):
      with tf.variable_scope('Model', reuse=None, initializer=initializer):
        train_input = KTHInput(config=config, data=train_data)
        model = BiLSTM(True, train_input, config)
      tf.scalar_summary("Training Loss", model.cost)
      tf.scalar_summary("Learning Rate", model.lr)

    sv = tf.train.Supervisor(logdir = FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        model.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
        train_perplexity = run_epoch(session, model, eval_op=model.train_op,
                       verbose=True)
        
      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
  tf.app.run()
