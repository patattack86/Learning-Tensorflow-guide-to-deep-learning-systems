#using abstractions in tensorflow to do linear regression. tensorflow.contrib

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target


NUM_STEPS = 200
MINIBATCH_SIZE = 506
#infer_real_valued_columns_from_input takes a matrix of n samples and n features and returns 
#a list of featurecolumn objects
#### This function is depreciated and will be removed in future versions, specify feature columns specifically
feature_columns = learn.infer_real_valued_columns_from_input(x_data)

reg = learn.LinearRegressor(feature_columns=feature_columns, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1))

reg.fit(x_data, boston.target, steps=NUM_STEPS,batch_size=MINIBATCH_SIZE)

MSE = reg.evaluate(x_data, boston.target, steps=1)

print(MSE)
