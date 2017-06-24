"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import video_train
import tensorflow as tf

from data.kth_data import KTHData

tf.app.flags.DEFINE_string("data_path", None,
               "Where the training/validation data is stored.")
tf.app.flags.DEFINE_string("save_path", None,
               "Model output directory.")
tf.app.flags.DEFINE_string("image_height", 120,
               "Model output directory.")
tf.app.flags.DEFINE_string("image_width", 160,
               "Model output directory.")
FLAGS = tf.app.flags.FLAGS


class Config(object):

  def __init__(self):
    """Configuration"""
    self.init_scale = 0.1
    self.learning_rate = 0.5
    self.max_grad_norm = 5
    self.num_layers = 2
    # num_steps: This value must be the same as the sequence_length value
    #  inside the data/convert_to_records.py when you generate the data.
    self.num_steps = 16
    self.hidden_size = 200
    self.max_epoch = 2
    self.max_max_epoch = 6
    self.keep_prob = 0.8
    self.lr_decay = 0.8
    self.batch_size = 20
    self.examples_per_shard = 23
    self.input_queue_memory_factor = 2


def main(_):
  """Main funtion to train the kth dtaset"""
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to KTH data directory")

  train_data = KTHData('train')
  assert train_data.data_files()

  config = Config()
  config.num_classes = train_data.num_classes()

  video_train.train(config, train_data)


if __name__ == '__main__':
  tf.app.run()
