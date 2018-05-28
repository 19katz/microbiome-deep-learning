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
from keras.layers.core import Dropout
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import itertools
from itertools import cycle, product

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
                    bias_initializer='zeros',activity_regularizer=regularizers.l1(10e-2))(input_img)

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


    return autoencoder, encoder, decoder



def create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation):
    
    model = Sequential()
    model.add(Dense(encoding_dim, activation=encoded_activation, input_dim=input_dim))
    model.add(Dense(1, activation=decoded_activation))

    # For a binary classification problem
              
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
 



def plot_confusion_matrix(cm, classes, file_name):
    
    cmap=pylab.cm.Reds
    """
    This function plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    pylab.figure()
    im = pylab.imshow(cm, interpolation='nearest', cmap=cmap)
    pylab.title('confusion_matrix')
    pylab.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pylab.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pylab.xlabel('True label')
    pylab.ylabel('Predicted label')
    pylab.gca().set_position((.1, 10, 0.8, .8))

    pylab.savefig(file_name , bbox_inches='tight')






def plot_roc_aucs(fpr, tpr, auc, acc,file_name):
    title='ROC Curves, auc=%s, acc=%s' %(auc,acc)
    pylab.figure()

    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title(title)
    #pylab.gca().set_position((.1, .7, .8, .8))


    pylab.savefig(file_name)
