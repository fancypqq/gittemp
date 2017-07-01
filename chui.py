import time
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import random

FEATURE_NUMBER = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUMBER_ITERATION = 1000
MOVING_AVERAGE_LENGTH = 5

def placeholder_inputs():
  feature_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                         FEATURE_NUMBER))
  labels_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
  return feature_placeholder, labels_placeholder


def fill_feed_dict(data, feature_pl, labels_pl):
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
    logits = tf.matmul(feature_pl, weights)
  return logits, weights

def get_loss(logits, prediction):
  loss = tf.reduce_mean(tf.square(logits-prediction))
  return loss

def get_train_op(loss, learning_rate):
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  train_op = optimizer.minimize(loss)
  return train_op


def run_training(dataset, origin_data):
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    feature_placeholder, labels_placeholder = placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits, weights = inference(feature_placeholder)

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
    feed_dict = fill_feed_dict(dataset[-BATCH_SIZE:,:], feature_placeholder,
                                 labels_placeholder)
    weights = sess.run(weights)
    variance = sess.run(loss, feed_dict=feed_dict)
    an = origin_data[-1, -1]
    an_1 = origin_data[-1, -2]
    an_2 = origin_data[-1, -3]
    mean=weights[-1]*an + weights[-2]*(an-an_1)+weights[-3]*(an+an_2-2*an_1)
    #print(dataset)
    print("Mean: "+str(mean))
    print("Variance: "+str(variance))
    

def dummy_sample():
  x=np.zeros((2,1000))
  x[0,:]=np.random.randn(1000)+100
  for i in range(1000):
      x[1][i]=1
  x[0][8]=150
  x[1][8]=0
  x[0][453]=60
  x[1][453]=0
  return x

def get_moving_average(data):
  length = data.shape[1]
  length = length - MOVING_AVERAGE_LENGTH
  averged_data = np.zeros((2, length))
  for i in range(length):
   for j in range(MOVING_AVERAGE_LENGTH):
     averged_data[:,i]=averged_data[:,i]+data[:,i+j]
   averged_data[0,i]=float(averged_data[0,i])/MOVING_AVERAGE_LENGTH
   if averged_data[1,i] != MOVING_AVERAGE_LENGTH:
     averged_data[1,i]=0
   else:
     averged_data[1,i]=1
  return averged_data

def prepare_training_data(data):
  length = data.shape[1]
  length = length - FEATURE_NUMBER
  y1=np.zeros((length, FEATURE_NUMBER+2))
  for j in range(length):
    y1[j, FEATURE_NUMBER+1]=1
  for k in range(length):
    y1[k, 0:FEATURE_NUMBER+1]=data[0, k:k+FEATURE_NUMBER+1]
  for k in range(length):
    y1[k, FEATURE_NUMBER+1]=sum(data[1,k:k+FEATURE_NUMBER+1])
    if y1[k, FEATURE_NUMBER+1] != FEATURE_NUMBER+1:
      y1[k, FEATURE_NUMBER+1]=0
    else:
      y1[k, FEATURE_NUMBER+1]=1
  y2=[]
  for k in range(length):
      if(y1[k,4]!=0):
          col=[y1[k][0],y1[k][1],y1[k][2],y1[k][3]]
          y2.append(col)
  z=[]
  for k in range(len(y2)):
      col=[y2[k][2]+y2[k][0]-2*y2[k][1],y2[k][2]-y2[k][1],y2[k][2],y2[k][3]]
      z.append(col)
  return np.array(z), np.array(y2)

def main():
  # TODO ADD DATA:
  # Because feature add label 
  data = dummy_sample()
  averged_data = get_moving_average(data)
  dataset, origin_data = prepare_training_data(averged_data)
  run_training(dataset, origin_data)


if __name__ == '__main__':
  main()