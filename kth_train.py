"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from bilstm_model import BiLSTM
from kth_data import KTHData

def main(_):
  train_dataset = KTHData('train')
  assert train_dataset.data_files()
  lstm_train.train(train_dataset)

if __name__ == '__main__':
  tf.app.run()
