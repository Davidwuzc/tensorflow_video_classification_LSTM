from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import video_processing as vp

class DataInput(object):
  """The input data."""

  def __init__(self, config, data):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = (data.num_examples_per_epoch() // batch_size) - 1
    # input_data size: [batch_size, num_steps]
    # targets size: [batch_size]
    self.input_data, self.targets, self.filenames = vp.distorted_inputs(
      data, config)

    # Data preprocessing: input_data
    #  string tensor [batch_size, num_steps] =>
    #    [batch_size, num_steps, height, width, channels]
    self.input_data = tf.map_fn(
      vp.decode_video, self.input_data, dtype=tf.float32)

    # Reference
    #  string tensor [batch_size, num_steps, height, width, channels] =>
    #    num_steps * [batch_size, height*width*channels]
    # self.input_data = tf.reshape(
    #   self.input_data, [batch_size, num_steps, -1])
    # self.input_data = [tf.squeeze(input_step, [1])
    #            for input_step in tf.split(1, num_steps, self.input_data)]