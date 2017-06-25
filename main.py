"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import video_train
import tensorflow as tf

from data.kth_data import KTHData
from data.lca_data import LCAData

tf.app.flags.DEFINE_string("data_path", None,
  "Where the training/validation data is stored.")
tf.app.flags.DEFINE_string("save_path", 'result',
  "Model output directory.")
tf.app.flags.DEFINE_string("dataset", 'KTH',
  "Select the dataset, default is KTH datasetk, choice between (KTH, LCA)")

FLAGS = tf.app.flags.FLAGS

config = {
  ''' Training parameters '''
  self.epoch                      = 6
  self.lr_decay                   = 0.8
  self.keep_prob                  = 0.8
  self.init_scale                 = 0.1 # weight initialization value (-init_scale, init_scale)
  self.batch_size                 = 20
  self.learning_rate              = 0.5
  self.max_grad_norm              = 5
  self.decay_begin_epoch          = 2
  self.examples_per_shard         = 23
  self.input_queue_memory_factor  = 2
  ''' Model parameters '''
  self.num_layers                 = 2
  # num_steps: This value must be the same as the sequence_length value
  #  inside the data/convert_to_records.py when you generate the data.
  self.num_steps                  = 16
  self.hidden_size                = 200
  ''' Image parameters '''
  self.image_width                = 160
  self.image_heigt                = 120
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
  
  assert train_data
  assert train_data.data_files()
  config.num_classes = train_data.num_classes()

  # Start training
  video_train.train(config, train_data)


if __name__ == '__main__':
  tf.app.run()
