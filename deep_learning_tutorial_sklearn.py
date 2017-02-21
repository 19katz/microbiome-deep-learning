# always run miniconda for keras:
# ./miniconda3/bin/python

import matplotlib  
matplotlib.use('Agg') 
import bz2
import numpy
from numpy import random
import pandas
import os
import pylab
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History 
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_autoencoder():

    # this is the size of our encoded representations
    encoding_dim = 5  
    # this is our input placeholder
    input_img = Input(shape=(5952,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(5952, activation='sigmoid')(encoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy']) # NOT SURE IF THIS IS THE RIGHT METRIC (SHOULD IT BE neg_mean_squared_error? MSE?)
    return autoencoder

autoencoder = KerasClassifier(build_fn=create_autoencoder, verbose=0)

#################################
# read in the data using pandas
#################################
filename="~/deep_learning/deep_learning_data/relative_abundance.txt.bz2"
rel_abundance_matrix = pandas.read_csv(filename, sep = '\t', index_col = 0).T 


###############
# create grid #
############### 

epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(batch_size=batches, nb_epoch=epochs)

grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid)
grid_result = grid.fit(rel_abundance_matrix.values, rel_abundance_matrix.values)

################################################
# analyze which parameters gave the best score #
################################################ 
 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

