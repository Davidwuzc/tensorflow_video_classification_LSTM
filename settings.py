"""configuration file for the project"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# video parameters
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('image_height', 160,
                            """Provide the height of the image.""")

tf.app.flags.DEFINE_integer('image_width', 120,
                            """Provide the height of the image.""")

tf.app.flags.DEFINE_integer('sequence_size', 5, 
                            """length of the video """
                            """proto""")

# image processing parameters
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 2,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, """
                            """e.g. 4, 2 or 1, if host memory is  """
                            """constrained. See comments in code for more """
                            """details.""")

# training parameters
tf.app.flags.DEFINE_string('train_dir', '/tmp/train_result',
                            """Directory where to write summary datas and""" 
                            """model checkpoints.""")

tf.app.flags.DEFINE_string('data_dir', '/tmp/sharded_data',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")

tf.app.flags.DEFINE_string('model', 'small',
                           """Model configuration. Possible options are: """
                           """small, medium, large.""")

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")