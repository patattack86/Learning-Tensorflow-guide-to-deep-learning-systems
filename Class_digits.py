#Classifcation on numbers dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#define constants
DATA_DIR = 'B:\Tensorflow books\Guide to tensorflow' 
NUM_STEPS = 1000 
MINIBATCH_SIZE = 100

#download and save data locally where Data_dir is where it will be stored
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

#variable element manipulated by computation
#placeholder supplied when triggering the argument, the image itself is a placeholder supplied by us
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

#represents the true and predicted labels
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

#defining the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits= y_pred, labels = y_true))

#Training the model with .5 learning rate
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))


#initialize all variables
with tf.Session() as sess:

    #model training
    sess.run(tf.global_variables_initializer())
    
    #training where we are taking 1,000 steps in the "right direction" as defined in the constant num_steps
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        #test, it's accuracy computing
        ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print ("Accuracy: {: .4}%".format(ans*100)
