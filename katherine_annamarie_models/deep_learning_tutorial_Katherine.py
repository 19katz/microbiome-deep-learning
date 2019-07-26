# always run miniconda for keras:
# ./miniconda3/bin/python

import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import bz2
import numpy as np
from numpy import random
import pandas
import os
import pylab
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K

backend = K.backend()

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

# variables for the activation functions
encoded_activation = 'relu'
decoded_activation = 'softmax'
# encoded_activation = 'linear'
# decoded_activation = 'linear'

# variable for user bias
bias = True

# "encoded" is the encoded representation of the input
# relu is apparently popular these days
encoded = Dense(encoding_dim, activation=encoded_activation, use_bias=bias,
                bias_initializer='zeros')(input_img)

# "decoded" is the lossy reconstruction of the input
# sigmoid used to give values between 0 and 1
decoded = Dense(5952, activation=decoded_activation, use_bias=bias,
                bias_initializer='zeros')(encoded)

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
#loss='mean_squared_error'
loss='kullback_leibler_divergence'
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # this makes sense when the input and output are binary
autoencoder.compile(optimizer='adadelta', loss=loss)
weightFile = os.environ['HOME'] + '/deep_learning_microbiome/model/weights.txt'
autoencoder.save_weights(weightFile)

#################
# Fit the model #
#################

cuts= np.array([300, 600, 900, 1200, 1500])
tests = np.subtract(1573, cuts)
allHistories = []
numEpochs = 50
batchSize = 256
for cut in cuts:
    x_train=rel_abundance_matrix.values[:cut]
    x_test=rel_abundance_matrix.values[cut:]
    history = History()
    autoencoder.load_weights(weightFile)
    autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[history])
    allHistories.append(history)

# history is a dictionary. To get the keys, type print(history.history.keys())

# how does loss change with number of epochs?
cut = 1300
x_train=rel_abundance_matrix.values[:cut]
x_test=rel_abundance_matrix.values[cut:]
history = History()
callbacks = [history]
if backend == 'tensorflow':
    callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

autoencoder.load_weights(weightFile)
autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=callbacks)

########################
# Make new predictions #
########################
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# file naming
testTrainInfo ='{}{}'.format('_numTrain-', x_train.shape[0]) + '{}{}'.format('_numTest-', x_test.shape[0])

#################
# Plots         #
#################
graph_dir = '~/deep_learning_microbiome/analysis/rel_abundance'

# file naming system
fileInfo = '{}{}'.format('_encDim-', encoding_dim) + '{}{}'.format('_bias-', bias) + '{}{}'.format('_numEpochs-', numEpochs) + '{}{}'.format('_batch-',  batchSize) + '_encActFunc-' + encoded_activation + '_decActFunc-' + decoded_activation + '_lossFunc-' + loss + '_backend-' + backend

# how does the loss change as the fraction of data in the test vs training change?
pylab.figure()
for hs in allHistories:
    pylab.plot(hs.history['val_loss'])
pylab.title('model loss by epochs and number of training samples')
pylab.ylabel('test loss')
pylab.xlabel('epoch')
pylab.legend(['300', '600','900','1200','1500'], loc='upper left')
pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .24, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .2, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/epoch_vs_loss_varied_training_samples'
                                 + fileInfo
                                 + '.pdf')
              , bbox_inches='tight')

pylab.figure()
pylab.plot(history.history['loss'])
pylab.plot(history.history['val_loss'])
pylab.title('model loss')
pylab.ylabel('training/test loss')
pylab.xlabel('epoch')
pylab.legend(['train', 'test'], loc='upper left')
pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs.')
pylab.figtext(0.02, .38, 'Backend: ' + backend)
pylab.figtext(0.02, .36, 'Loss function: ' + loss)
pylab.figtext(0.02, .32, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .28, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .24, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .2, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .16, 'Number of training samples: {}'.format(x_train.shape[0]))
pylab.figtext(0.02, .12, 'Number of test samples: {}'.format(x_test.shape[0]))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/epoch_vs_training_test_loss'
                                 + fileInfo
                                 + testTrainInfo
                                 + '.pdf'),
              bbox_inches='tight')




# how do the points before and after encoding compare (can we make a y=x line?)
#matplotlib.rcParams['agg.path.chunksize'] = 10000 # might need this as there are too many points.
pylab.figure()
num_data_pts=len(x_test.flatten())
indexes=np.random.choice(num_data_pts,1000,replace=False)
input_data=x_test.flatten()[indexes]
decoded_data=decoded_imgs.flatten()[indexes]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('input vs decoded data')

ax.set_xlabel('input data')
ax.set_ylabel('decoded data')

ax.scatter(input_data,decoded_data, s = 1)
pylab.gca().set_position((.1, .6, .8, .6))

pylab.figtext(0.02, .4, 'This graph takes 1000 randomly selected values of the input data \n and the corresponding 1000 values of the decoded data \n and graphs a scatter plot of them.')
pylab.figtext(0.02, .38, 'Backend: ' + backend)
pylab.figtext(0.02, .36, 'Loss function: ' + loss)
pylab.figtext(0.02, .32, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .28, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .24, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .2, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .16, 'Number of training samples: {}'.format(x_train.shape[0]))
pylab.figtext(0.02, .12, 'Number of test samples: {}'.format(x_test.shape[0]))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)

plt.show()
pylab.savefig(os.path.expanduser(graph_dir + '/data_decoded'
                                 + fileInfo
                                 + testTrainInfo
                                 + '.pdf')
              , bbox_inches='tight')


