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
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
    return autoencoder

def create_encoder():
    input_img = Input(shape=(5952,))  
    # create encoder
    encoder = Model(input=input_img, output=encoded)
    return encoder

def create_decoder(autoencoder): # this doesn't work
    # create decoder
    encoding_dim = 5
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    return decoder
    

autoencoder = KerasClassifier(build_fn=create_autoencoder, verbose=0)
#encoder=create_encoder()
#decoder=create_decoder(autoencoder)




#################################
# read in the data using pandas
#################################
filename="~/deep_learning/deep_learning_data/relative_abundance.txt.bz2"
rel_abundance_matrix = pandas.read_csv(filename, sep = '\t', index_col = 0).T 

# change to using scikit learn for 10-fold validation
x_train=rel_abundance_matrix.values[:1300]
x_test=rel_abundance_matrix.values[1300:]


###############
# create grid #
############### 

epochs = [50, 100, 150]
batches = [5, 10, 20]
grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid)  

grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid)
grid_result = grid.fit(x_train, x_train)


#################
# Fit the model #
#################

x_train=rel_abundance_matrix.values[:1300]
x_test=rel_abundance_matrix.values[1300:]   
history = History() 
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[history])


####################
# Make predictions #
####################
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#################
# Plots         #
#################

# how does loss change with number of epochs?
pylab.figure()
pylab.plot(history.history['loss'])
pylab.plot(history.history['val_loss'])
pylab.title('model loss')
pylab.ylabel('loss')
pylab.xlabel('epoch')
pylab.legend(['train', 'test'], loc='upper left')
pylab.savefig(os.path.expanduser('~/deep_learning/deep_learning_analysis/epoch_vs_loss.pdf'), bbox_inches='tight')

# how does the loss change as the fraction of data in the test vs training change?

# how do the points before and after encoding compare (can we make a y=x line?)
#matplotlib.rcParams['agg.path.chunksize'] = 10000 # might need this as there are too many points. 
pylab.figure()
num_data_pts=len(x_test.flatten())
indexes=numpy.random.choice(num_data_pts,1000,replace=False)
input_data=x_test.flatten()[indexes]
decoded_data=decoded_imgs.flatten()[indexes]
pylab.plot(input_data, decoded_data)
pylab.title("input vs decoded data")
pylab.ylabel("decoded data")
pylab.xlabel("input data")
pylab.savefig(os.path.expanduser('~/deep_learning/deep_learning_analysis/data_decoded.pdf'), bbox_inches='tight')

# Plot alpha diversity before and after encoding

# plot beta diveristy before and after encoding



