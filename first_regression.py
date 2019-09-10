import tensorflow as tf
import numpy as np

###----------Chapter three example----------####

#simple regression

#step one is to create the placeholder values for both x an dy
x = tf.placeholder(tf.float32, shape=[None, 3])
y_true = tf.placeholder(tf.float32, shape=None)

#w represents the weights in our model and b represents the intercept
w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
b = tf.Variable(0,dtype=tf.float32,name='bias')

#now we define our model which is just a simple linear regression, matmul of x and w plus bias term b
y_pred = tf.matmul(w,tf.transpose(x)) + b

#now create our loss function, first example is MSE
los = tf.reduce_mean(tf.square(y_true-y_pred))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)
