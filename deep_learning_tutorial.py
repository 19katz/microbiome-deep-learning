# always run miniconda for keras:
# ./miniconda3/bin/python

import bz2
import numpy
import pandas
import os
from keras.layers import Input, Dense
from keras.models import Model

#################################
# read in the data using pandas
#################################
filename=os.path.expanduser("~/deep_learning/relative_abundance.txt.bz2")
rel_abundance_matrix = pandas.read_csv(filename, sep = '\t', index_col = 0)
rel_abundance_matrix.head()
rel_abundance_matrix.values


################################
# set up a model (autoencoder)
################################

# this is the size of our encoded representations -- NEED TO PLAY AROUND WITH THIS VARIABLE
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
# 1573 comes from the number of columns in our input data
# the number of rows is the number of data points we have
input_img = Input(shape=(1573,))

# "encoded" is the encoded representation of the input
# relu is apparently popular these days
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
# sigmoid used to give values between 0 and 1
decoded = Dense(1573, activation='sigmoid')(encoded)

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
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#################
# Fit the model #
#################

x_train=rel_abundance_matrix.values
x_test=rel_abundance_matrix.values     
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

####################
# Make predictions #
####################
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
