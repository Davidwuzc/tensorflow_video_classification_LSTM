"""A binary to train BiLSTM on the KTH data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from bilstm_model import BiLSTM
from kth_data import KTHData
import video_processing as vp

tf.app.flags.DEFINE_string("data_path", None,
          "Where the training/validation data is stored.")
tf.app.flags.DEFINE_string("save_path", None,
          "Model output directory.")
FLAGS = tf.app.flags.FLAGS


class Config(object):
  """Configuration"""
  init_scale = 0.1
  learning_rate = 0.5
  max_grad_norm = 5
  num_layers = 2
  # num_steps: This value must be the same as the sequence_length value 
  #  inside the data/convert_to_records.py when you generate the data.
  num_steps = 16
  hidden_size = 200
  max_epoch = 2
  max_max_epoch = 6
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 20
  examples_per_shard = 23
  input_queue_memory_factor = 2

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
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


class KTHInput(object):
  """The input data."""
  def __init__(self, config, data):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = (data.num_examples_per_epoch() // batch_size) - 1
    # input_data size: [batch_size, num_steps]
    # targets size: [batch_size]
    self.input_data, self.targets, _ = vp.distorted_inputs(data, config)

    # Data preprocessing: input_data
    #  string tensor [batch_size, num_steps] => 
    #    [batch_size, num_steps, height, width, channels]
    self.input_data = tf.map_fn(decode_jpeg, self.input_data)
    self.input_data = tf.reshape(self.input_data, [batch_size, num_steps, -1])
    self.input_data = [tf.squeeze(input_step, [1])
      for input_step in tf.split(1, num_steps, self.input_data)]

def run_epoch(self, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0

  fetches = {
    "cost": model.cost,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    vals = session.run(fetches)
    cost = vals["cost"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f -- perplexity: %.3f -- speed: %.0f wps" %
        (step*1.0 / model.input.epoch_size,
        np.exp(costs / iters),
        iters*model.input.batch_size / (time.time()-start_time)))

  return np.exp(costs / iters)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to KTH data directory")

  train_data = KTHData('train')
  assert train_data.data_files()
  
  config = Config()
  config.num_classes = train_data.num_classes()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                          config.init_scale)
    with tf.name_scope('Train'):
      with tf.variable_scope('Model', reuse=None, initializer=initializer):
        train_input = KTHInput(config=config, data=train_data)
        model = BiLSTM(True, train_input, config)
      tf.scalar_summary("Training Loss", model.cost)
      tf.scalar_summary("Learning Rate", model.lr)

    sv = tf.train.Supervisor(logdir = FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        model.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
        train_perplexity = run_epoch(session, model, eval_op=model.train_op,
                       verbose=True)
        
      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
  tf.app.run()
