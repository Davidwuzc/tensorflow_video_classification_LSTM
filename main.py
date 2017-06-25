"""A binary to train BiLSTM on the UCF101 data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import video_train
import tensorflow as tf
from data.ucf101_data import UCF101Data
from data.kth_data    import KTHData
from data.lca_data    import LCAData

tf.app.flags.DEFINE_string("data_path", None,
                           "Where the training/validation data is stored.")
tf.app.flags.DEFINE_string("save_path", 'result',
                           "Model output directory.")
tf.app.flags.DEFINE_string("dataset", 'KTH',
                           "Select the dataset, default is KTH datasetk, \
                           choice between (KTH, LCA, UCF)")
FLAGS = tf.app.flags.FLAGS

config = {
  ''' Training parameters '''
  epoch                      = 6
  lr_decay                   = 0.8
  keep_prob                  = 0.8
  init_scale                 = 0.1 # weight initialization value (-init_scale, init_scale)
  batch_size                 = 20
  # (num_steps % c3d_num_steps) must equal to 0
  c3d_num_steps              = 9
  learning_rate              = 0.5
  max_grad_norm              = 5
  decay_begin_epoch          = 2
  examples_per_shard         = 23
  input_queue_memory_factor  = 2
  ''' Model parameters '''
  num_layers                 = 2
  # num_steps: This value must be the same as the sequence_length value
  #  inside the data/convert_to_records.py when you generate the data.
  num_steps                  = 108
  hidden_size                = 200
  ''' Image parameters '''
  image_width                = 160
  image_heigt                = 120

  # C3D model parameters
  c3d_weights = {
    'wc1': [3, 3, 3, 3, 7],
    'wc2': [3, 3, 3, 7, 14],
    'wc3a': [3, 3, 3, 14, 28],
    'wc3b': [3, 3, 3, 28, 28],
    'wc4a': [3, 3, 3, 28, 56],
    'wc4b': [3, 3, 3, 56, 56],
    'wc5a':  [3, 3, 3, 56, 56],
    'wc5b':  [3, 3, 3, 56, 56],
    'wd1': [896, 120]
  }

  c3d_biases = {
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
}

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to KTH data directory")

  # Select the dataset
  train_data = None
  if FLAGS.dataset == 'KTH'
    train_data = KTHData('train')
  elif FLAGS.dataset == 'LCA'
    train_data = LCAData('train')
  elif FLAGS.dataset == 'UCF'
    train_data = UCF101Data('train')
  
  assert train_data
  assert train_data.data_files()
  config.num_classes = train_data.num_classes()

  # Start training
  video_train.train(config, train_data)

if __name__ == '__main__':
  tf.app.run()
