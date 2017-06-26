from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .video_processing import distorted_inputs

FLAGS = tf.app.flags.FLAGS

def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.channels)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.image_height, FLAGS.image_width],
        align_corners=False)
    image = tf.squeeze(image, [0])
  return image


def decode_video(video_buffer):
  """Decode list of string Tensor into list of 3-D float image Tensor.
  Args:
    video_buffer: tensor, shape [num_steps].
  Returns:
    list of 3-D float Tensor with values ranging from [0, 1).
  """
  # Decode the images of one video
  return tf.map_fn(decode_jpeg, video_buffer, dtype=tf.float32)


class DataInput(object):
  """The input data."""

  def __init__(self, config, data):
    self.batch_size = batch_size = config['batch_size']
    self.num_steps  = num_steps = config['num_steps']
    self.epoch_size = (data.num_examples_per_epoch() // batch_size) - 1
    # input_data size: [batch_size, num_steps]
    # targets size: [batch_size]
    self.input_data, self.targets, self.filenames = distorted_inputs(
      data, config)

    # Data preprocessing: input_data
    #  string tensor [batch_size, num_steps] =>
    #    [batch_size, num_steps, height, width, channels]
    self.input_data = tf.map_fn(decode_video, self.input_data, 
                                dtype=tf.float32)