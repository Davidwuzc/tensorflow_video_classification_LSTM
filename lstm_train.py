from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path
from datetime import datetime
import numpy as np

import video_processing
import cnn
import settings

FLAGS = tf.app.flags.FLAGS

# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
class SmallConfig(object):
    """ Small config."""
    # Parameters
    learning_rate = 0.1
    training_iters = 10000
    batch_size = FLAGS.batch_size
    display_step = 1
    # LSTM Network parameters
    num_input = 2048
    num_steps = FLAGS.sequence_size
    num_hidden = 256 # hidden layer number of features
    num_layer = 3

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

# ----------------------------------------------------------------------------
# LSTM Model
# ----------------------------------------------------------------------------
def BiLSTM(x, weights, biases):
    """ Bidirectional LSTM neural network.

    Use this function to create the bidirection LSTM nerual network model

    Args:
        x: a tensor placeholder that represent batches of video
        weight: variable, all the weight variable of the model
        biases: variable, all the biases variable of the model

    Returns:
        pred: tensor. predition value calculated by the lastest model
    """
    # get the configuration
    config = get_config()

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_step, n_input)
    # Required shape: 'num_steps' tensors list of shape (batch_size, n_input)

    with tf.name_scope('data_preprocessing'):
        # Permuting batch_size and n_step
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_step * batch_size, num_input)
        x = tf.reshape(x, [-1, config.num_input])
        # Split to get a list of 'num_steps' tensors of shape 
        #   (batch_size, num_input)
        x = tf.split(0, config.num_steps, x)

    with tf.name_scope('BiLSTM_cells'):
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_hidden,
                                                    state_is_tuple=True)
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layer, 
                                            state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_hidden,
                                                    state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layer, 
                                            state_is_tuple=True)
        
    with tf.name_scope('Bidrectional_rnn'):
        # Get lstm cell output
        outputs, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, x,
                                                dtype=tf.float32)

    with tf.name_scope('activation'):
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights) + biases

# ----------------------------------------------------------------------------
# CNN Model
# ----------------------------------------------------------------------------
def cnn_build():
    """build the cnn graph using prebuild inception model"""
    cnn.maybe_download_and_extract()
    # Creates graph from saved GraphDef.
    cnn.create_graph()

# ----------------------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------------------
def train(dataset):
    # get the configuration settings
    config = get_config()
    num_classes = dataset.num_classes()

    with tf.name_scope('input'):
        # tf Graph image inputs and logits input
        x = tf.placeholder(tf.float32,
        [None, config.num_steps, config.num_input], 
        name='x-input')
        y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.random_normal([2*config.num_hidden, num_classes]))
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.random_normal([num_classes]))
    with tf.name_scope('BiLSTM'):
        pred = BiLSTM(x, weights, biases)
    with tf.name_scope('cost'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # coordinator for controlling queue threads
    coord = tf.train.Coordinator()
    
    # build the cnn feature extractor graph first
    cnn_build()

    # initialize the image and label operator
    videos_op, labels_op, _ = video_processing.distorted_inputs(
        dataset,
        batch_size=config.batch_size)
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Create a saver to recurrently save all the vaiables 
    saver = tf.train.Saver(tf.all_variables())

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # restore the model from the checkpoint
        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # start all the queue thread
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Merge all the summary and write then out to the summary folder 
        merged_summaries = tf.merge_all_summaries()
        # Clear the summary data folder beforehand
        summary_path = os.path.join(FLAGS.train_dir, 'summary')
        if tf.gfile.Exists(summary_path):
            tf.gfile.DeleteRecursively(summary_path)
            tf.gfile.MakeDirs(summary_path)
            writer = tf.train.SummaryWriter(summary_path, graph=sess.graph)

        # image feature operator
        feature_op = sess.graph.get_tensor_by_name('pool_3:0')

        # Keep training until reach max iterations
        for step in range(config.training_iters):
            # get the image and label data
            summary_result, videos, labels = sess.run([
                merged_summaries, videos_op, 
                labels_op])

            videos_features = []
            for video in videos:
                images_features = []
                for image in video:
                    image_features = sess.run(feature_op, 
                                              {'DecodeJpeg/contents:0': image})
                    images_features.append(np.squeeze(image_features))
                videos_features.append(images_features)

            # run the optimizer
            sess.run(optimizer, feed_dict={x: videos_features, y: labels})
            # write the summary result to the writer
            writer.add_summary(summary_result, step)

            # print out the loss and accuracy
            if step % config.display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: videos_features, y: labels})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: videos_features, y: labels})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == config.training_iters:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # request to stop the input queue
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
    # close the summary writer
    writer.close()
    print("Optimization Finished!")