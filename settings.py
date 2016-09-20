"""configuration file for the project"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# video parameters
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('sequence_size', 40, 
                            """ Size of the images size in one example """
                            """proto""")

# image processing parameters
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, """
                            """e.g. 4, 2 or 1, if host memory is  """
                            """constrained. See comments in code for more """
                            """details.""")

# training parameters
tf.app.flags.DEFINE_string('train_dir', '/Volumes/passport/datasets/action_LCA/video_data/checkpoint',
                            """Directory where to write event logs and""" 
                            """checkpoint."""
)
tf.app.flags.DEFINE_string('model', 'small',
                            """Model configuration. Possible options are: """
                            """small, medium, large."""
)
tf.app.flags.DEFINE_string('data_dir', '/Volumes/passport/datasets/action_LCA/video_data/sharded_data',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
