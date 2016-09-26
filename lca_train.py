"""A binary to train BiLSTM on the LCA data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import lstm_train
from lca_data import LCAData

FLAGS = tf.app.flags.FLAGS

def main(_):
  train_dataset = LCAData('train')
  assert train_dataset.data_files()
  # if tf.gfile.Exists(FLAGS.train_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
  # tf.gfile.MakeDirs(FLAGS.train_dir)
  lstm_train.train(train_dataset)

if __name__ == '__main__':
  tf.app.run()
