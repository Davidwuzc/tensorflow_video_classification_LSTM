"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kth_data import KTHData
import video_train

tf.app.flags.DEFINE_string("data_path", None,
               "Where the training/validation data is stored.")
tf.app.flags.DEFINE_string("save_path", None,
               "Model output directory.")
tf.app.flags.DEFINE_string("image_height", 112,
               "Model output directory.")
tf.app.flags.DEFINE_string("image_width", 112,
               "Model output directory.")
tf.app.flags.DEFINE_string("image_channel", 3,
               "Model output directory.")
FLAGS = tf.app.flags.FLAGS


class Config(object):

  def __init__(self):
    """Configuration"""
    self.init_scale = 0.1
    self.learning_rate = 0.01
    self.max_grad_norm = 5
    self.num_layers = 2
    # num_steps: This value must be the same as the sequence_length value
    #  inside the data/convert_to_records.py when you generate the data.
    self.num_steps = 108
    # (num_steps % c3d_num_steps) must equal to 0
    self.c3d_num_steps = 9
    self.hidden_size = 50
    self.max_epoch = 2
    self.max_max_epoch = 6
    self.keep_prob = 0.8
    self.lr_decay = 0.8
    self.batch_size = 20
    self.examples_per_shard = 23
    self.input_queue_memory_factor = 2  

    # C3D parameters
    self.c3d_weights = {
      'wc1': [3, 3, 3, FLAGS.image_channel, 7],
      'wc2': [3, 3, 3, 7, 14],
      'wc3a': [3, 3, 3, 14, 28],
      'wc3b': [3, 3, 3, 28, 28],
      'wc4a': [3, 3, 3, 28, 56],
      'wc4b': [3, 3, 3, 56, 56],
      'wc5a':  [3, 3, 3, 56, 56],
      'wc5b':  [3, 3, 3, 56, 56],
      'wd1': [896, 200]
    }
    self.c3d_biases = {
      'bc1': [7],
      'bc2': [14],
      'bc3a': [28],
      'bc3b': [28],
      'bc4a': [56],
      'bc4b': [56],
      'bc5a': [56],
      'bc5b': [56],
      'bd1': [120]
    }
    


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
