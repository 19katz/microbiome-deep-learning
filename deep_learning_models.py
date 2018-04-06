# always run miniconda for keras:
# ./miniconda3/bin/python

import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import bz2
import numpy as np
from numpy import random
import pandas as pd
import os
import pylab
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K

backend = K.backend()

def create_autoencoder(encoding_dim, input_dim, encoded_activation, decoded_activation):

    ################################
    # set up a model (autoencoder)
    ################################

    # encoding_dim is the size of our encoded representations 
    #input_dim is the number of kmers (or columns) in our input data
    input_img = Input(shape=(input_dim,))


    # variable for user bias
    bias = True

    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation=encoded_activation, use_bias=bias,
                bias_initializer='zeros')(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation=decoded_activation, use_bias=bias,
                bias_initializer='zeros')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_img, outputs=decoded)

    ##############################################
    #Let's also create a separate encoder model:
    ##############################################

    # this model maps an input to its encoded representation
    encoder = Model(inputs=input_img, outputs=encoded)

    ###############################
    # As well as the decoder model:
    ################################

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))


    ######################
    # compile the encoder
    ######################
    #loss='mean_squared_error'
    loss='kullback_leibler_divergence'
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # this makes sense when the input and output are binary
    autoencoder.compile(optimizer='adadelta', loss=loss)
    #weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'
    #autoencoder.save_weights(weightFile)


    return autoencoder
