#using abstractions in tensorflow to do linear regression. tensorflow.contrib

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

#infer_real_valued_columns_from_input takes a matrix of n samples and n features and returns 
#a list of featurecolumn objects
feature_columns = learn.infer_real_valued_columns_from_input(x_data)

