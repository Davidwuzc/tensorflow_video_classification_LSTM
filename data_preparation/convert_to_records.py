"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/run/video1/00001.jpeg
  data_dir/run/video1/00002.jpeg
  data_dir/run/video1/00003.jpeg
  ...
  data_dir/run/video2/00001.jpeg
  data_dir/run/video2/00002.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 64 and 8 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  raw/image/001: 
  ...
  raw/image/nnn: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'
  image/filename: string containing the basename of the image file
            e.g. '00001.JPEG' or '00002.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'walk'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string('train_directory', '/tmp/dataset/train_directory',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/dataset/train_directory',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/dataset/result',
                           'Output data directory')
tf.app.flags.DEFINE_string('label_file', '/tmp/dataset/label.txt', 'Labels file')

tf.app.flags.DEFINE_integer('train_shards', 64,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('sequence_length', 16,
                            'The length of one video clips ')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_boolean('sequence_random', True,
                            'Determine whether to shuffle the image order or not.')


FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(foldername, images_buffer, label, text, height, width):
  """Build an Example proto for an example.

  Args:
    foldername: string, path to an image file, e.g., '/training_data/walk/video1'
    images_buffer: list, containing string of JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  # create the feature data for the TFRecord example
  images = {}
  for index, image in enumerate(images_buffer):
    images['raw/image/%03d' % index] = _bytes_feature(image)

  feature_dict = {
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/colorspace': _bytes_feature(colorspace),
    'image/channels': _int64_feature(channels),
    'image/class/label': _int64_feature(label),
    'image/class/text': _bytes_feature(text),
    'image/format': _bytes_feature(image_format),
    'image/filename': _bytes_feature(os.path.basename(foldername)),
  }
  
  feature_dict.update(images)

  # create the TFRecord Example
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(foldername, coder):
  """Process a single image file.

  Args:
    foldernames: string, path to a video folder e.g., '/path/to/video'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    videos_buffer: list, contains list of video with specific sequence length. These video is actually list of strings of JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  videos_data = []
  images_data = []
  filenames = tf.gfile.Glob(foldername + '/*')

  count = 0
  for filename in filenames:
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
      # print('Converting PNG to JPEG for %s' % filename)
      image_data = coder.png_to_jpeg(image_data)
    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    # Add the image to the images data
    images_data.append(image_data)
    count += 1
    if count % FLAGS.sequence_length == 0:
      videos_data.append(images_data)

  if len(videos_data) == 0:
    raise ValueError('sequence length is too long, please set the length smaller than the video length')
  return videos_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, foldernames,
                               texts, labels, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    foldernames: list of strings; each string is a path to a video file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      foldername = foldernames[i]
      label = labels[i]
      text = texts[i]

      videos_buffer, height, width = _process_image(foldername, coder)

      for images_buffer in videos_buffer:
        example = _convert_to_example(foldername, images_buffer, label,
                                      text, height, width)
        writer.write(example.SerializeToString())
        counter += 1
        shard_counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d videos in thread batch.' %
              (datetime.now(), thread_index, counter))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d videos' %
          (datetime.now(), thread_index, shard_counter))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d videos in total' %
        (datetime.now(), thread_index, counter))
  sys.stdout.flush()


def _process_image_files(name, foldernames, texts, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    foldernames: list of strings; each string is a path to a video folder
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(foldernames) == len(texts)
  assert len(foldernames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(foldernames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, foldernames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)

def _find_video_folders(data_dir, label_file):
  """Build a list of all video folders and labels in the data set.

  Args:
    data_dir: string, path to the root directory of video folders.

      Assumes that the video data set resides in JPEG files located in
      the following directory structure.

        data_dir/walk/video1/00001.JPEG
        data_dir/walk/video1/00002.JPEG
        ...
        data_dir/walk/video2/00001.jpg
        ...

      where 'walk' is the label associated with these images.
      number 1..n means that all the images in folder video1 belongs to one video

    label_file: string, path to the label file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        walk
        run
        play
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    folders: list of strings; each string is a path to an video folder.
    texts: list of strings; each string is the class, e.g. 'walk'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      label_file, 'r').readlines()]

  labels = []
  folders = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of video files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    folders.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all video folder in order to guarantee
  # random ordering of the videos with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  if FLAGS.sequence_random:
    shuffled_index = range(len(folders))
    random.seed(12345)
    random.shuffle(shuffled_index)

    folders = [folders[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

  print('Found %d video files across %d labels inside %s.' %
        (len(folders), len(unique_labels), data_dir))
  return folders, texts, labels


def _process_dataset(name, directory, num_shards, label_file):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    label_file: string, path to the labels file.
  """
  foldernames, texts, labels = _find_video_folders(directory, label_file)
  _process_image_files(name, foldernames, texts, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  #_process_dataset('validation', FLAGS.validation_directory,
  #                 FLAGS.validation_shards, FLAGS.label_file)
  _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.label_file)


if __name__ == '__main__':
  tf.app.run()
