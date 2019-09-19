import tensorflow as tf

###----------Chapter four example----------####

#Define helper functions which are used to create layers

def weight_variable(shape):
    #outputs random values from a truncated normal distribution, probability distribution from a normal distribution
    #we're defininf the weights for conncected or convolutional layers
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #defininf the bias element in connected or convoluted network, initiliazed with constant of .1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #specifies the convolution we're using
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #sets the max pool size to half the size across H/W dimension and quarter size of feature map
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    #Linear convolution used followed by ReLU nonlnearity
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size. size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
