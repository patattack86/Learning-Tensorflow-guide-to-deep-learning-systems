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

#now ready to set up model with our layers defined. 

#define placeholders for images and correct labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#reshape the image data into 2D format with 28x28x1 size
x_image = tf.reshape(x, [-1, 28, 28, 1])

#two layers of convolution and pooling witrh 5x5 convolutions
conv1 = conv_layer(x_iamge, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

#Size of image now reduced to 7x7xc64, 64 meaning number of feature mapes created in second convolution
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

#output is fully connected layer with 10 units corresponding to number of labels in dataset, with minst dataset there is a possible of 10 labels.
y_conv = full_layer(full1_drop, 10)
