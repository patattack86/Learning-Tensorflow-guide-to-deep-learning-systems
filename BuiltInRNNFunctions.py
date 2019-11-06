import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

element_size = 28;time_steps = 28;num_classes = 10
batch_size = 128;hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32,shape=[None, time_steps,element_size],name='inputs')

y = tf.placeholder(tf.float32, shape = [None, num_classes], name='inputs')

#using tensorflow built in function...so much fucking easier
rnn_cell = tf.contrib.rnn.BasiBasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_rnn_cell, _inputs, NotImplementedType=tf.float32)
