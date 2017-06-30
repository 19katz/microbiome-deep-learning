# always run miniconda for keras:
# ./miniconda3/bin/python

import matplotlib  
matplotlib.use('Agg') # this suppresses the console for plotting 
import bz2
import numpy
from numpy import random
import pandas
import os
import pylab
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History 

#################################
# read in the data using pandas
#################################
filename="~/deep_learning_microbiome/data/relative_abundance.txt.bz2"
rel_abundance_matrix = pandas.read_csv(filename, sep = '\t', index_col = 0).T # note that .T transposes permanently

#other operations to take note of:
#rel_abundance_matrix.head()
#rel_abundance_matrix.values # gives numpy representation
#rel_abundance_matrix.values.transpose()

################################
# set up a model (autoencoder)
################################

# this is the size of our encoded representations -- NEED TO PLAY AROUND WITH THIS VARIABLE
encoding_dim = 3  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
# 1573 comes from the number of columns in our input data
# the number of rows is the number of data points we have
input_img = Input(shape=(5952,))

# "encoded" is the encoded representation of the input
# relu is apparently popular these days
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
# sigmoid used to give values between 0 and 1
decoded = Dense(5952, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

##############################################  
#Let's also create a separate encoder model:
############################################## 

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

###############################
# As well as the decoder model:
################################

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


######################
# compile the encoder
######################
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # this makes sense when the input and output are binary
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')


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

# history is a dictionary. To get the keys, type print(history.history.keys())

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
pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/epoch_vs_loss.pdf'), bbox_inches='tight')

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
pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/data_decoded.pdf'), bbox_inches='tight')

# Plot alpha diversity before and after encoding

# plot beta diveristy before and after encoding



