#understanding the basics of creating graphs and then implementing computations using tf.sess()

import tensorflow as tf

###----------Chapter three example one----------####

#start by creating the graph
a = tf.constant(5) 
b = tf.constant(2) 
c = tf.constant(3)

d = tf.multiply(a,b) 
e = tf.add(c,b) 
f = tf.subtract(d,e)

##Run the computations of the graph using session
sess = tf.Session() 
outs = sess.run(f) 
sess.close() 
print("outs = {}".format(outs))

###----------Chapter three example two----------####

#Create the graph
a = tf.constant(5) 
b = tf.constant(2) 

c = tf.add(a,b) 
d = tf.multiply(a,b) 
e = tf.subtract(d,c)
f = tf.add(d,c) 
g = tf.divide(f,e)


#run the computations
sess = tf.Session() 
outs = sess.run(g) 
sess.close() 
print("outs = {}".format(outs))
