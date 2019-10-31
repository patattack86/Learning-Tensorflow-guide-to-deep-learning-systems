import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r"B:\Tensorflow books\Guide to tensorflow\scripts", one_hot=True)

#parameters

#dimension of each vector in our sequence
element_size = 28
#number of elements in a sequence
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

#saving tensorboard summary
Log_Dir = "logs/RNN_with_summaries"

#Create placeholder for input and label
inputs = tf.placeholder(tf.float32, shape =[None, time_steps, element_size], name='inputs')

y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

batch_x, batch_y = mnist.train.next_batch(batch_size)
# Reshape data to get 28 sequences of 28 pixel
batch_x = batch_x.reshape((batch_size, time_steps, element_size))

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev', stddev):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#create weights and biases for inputs and hidden layers
