
import time
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import random

FEATURE_NUMBER = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUMBER_ITERATION = 10

def placeholder_inputs():
  feature_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                         FEATURE_NUMBER))
  labels_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
  return feature_placeholder, labels_placeholder


def fill_feed_dict(data, feature_pl, labels_pl):
  # TODO
  feed_dict = {
      feature_pl: data[:,:-1],
      labels_pl: data[:,-1],
  }
  return feed_dict


def inference(feature_pl):
  with tf.name_scope('linear_regression'):
    weights = tf.Variable(
        tf.truncated_normal([FEATURE_NUMBER, 1],
                            stddev=1.0 / math.sqrt(float(FEATURE_NUMBER))),
        name='weights')
    biases = tf.Variable(tf.zeros([1]),
                         name='biases')
    logits = tf.matmul(feature_pl, weights) + biases
  return logits

def get_loss(logits, labels):
  loss = tf.reduce_mean(tf.square(logits-labels))
  return loss

def get_train_op(loss, learning_rate):
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  train_op = optimizer.minimize(loss)
  return train_op

def run_training(dataset):
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    feature_placeholder, labels_placeholder = placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits = inference(feature_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss = get_loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = get_train_op(loss, LEARNING_RATE)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # And then after everything is built, start the training loop.

    for step in xrange(NUMBER_ITERATION):

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      data = np.array(random.sample(dataset, BATCH_SIZE))
      feed_dict = fill_feed_dict(data, feature_placeholder,
                                 labels_placeholder)

      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
      print('haha')


def main():
  # TODO ADD DATA:
  # Because feature add label 
  dataset = np.random.randn(5000, FEATURE_NUMBER+1)
  run_training(dataset)


if __name__ == '__main__':
  main()