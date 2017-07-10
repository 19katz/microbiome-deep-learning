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
encoded = Dense(encoding_dim, activation='relu', use_bias=False,
                bias_initializer='zeros')(input_img)

# "decoded" is the lossy reconstruction of the input
# sigmoid used to give values between 0 and 1
decoded = Dense(5952, activation='sigmoid', use_bias=False,
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
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # this makes sense when the input and output are binary
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
weightFile = "/Users/KatherineZhang/deep_learning_microbiome/model/weights.txt"
autoencoder.save_weights(weightFile)

#################
# Fit the model #
#################

cut = 0
a = np.array([1, 2, 3, 4, 5])
allHistories = []
for ind in a:
    cut += 300
    x_train=rel_abundance_matrix.values[:cut]
    x_test=rel_abundance_matrix.values[cut:]
    history = History()
    autoencoder.load_weights(weightFile)
    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[history])
    allHistories.append(history)

# history is a dictionary. To get the keys, type print(history.history.keys())

####################
# Make predictions #
####################
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
print(decoded_imgs)

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
pylab.figure()
for ind in a:
    pylab.plot(allHistories[ind - 1].history['val_loss'])
pylab.title('model loss by epochs and number of training samples')
pylab.ylabel('test loss')
pylab.xlabel('epoch')
pylab.legend(['300', '600','900','1200','1500'], loc='upper left')
pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/epoch_vs_loss_varied_training_samples.pdf'), bbox_inches='tight')

# error per sample when the model is trained by 1300 samples
'''cut = 1300
    x_train=rel_abundance_matrix.values[:cut]
    x_test=rel_abundance_matrix.values[cut:]
    history = History()
    autoencoder.fit(x_train, x_train,
    nb_epoch=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[history])
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    pylab.figure()
    
    print(list(map(np.linalg.norm, decoded_imgs)))
    print(list(map(np.linalg.norm, x_test - decoded_imgs)))
    print(list(map(np.linalg.norm, x_test)))
    
    error = np.divide(list(map(np.linalg.norm, x_test - decoded_imgs)), list(map(np.linalg.norm, x_test)))
    pylab.plot(np.arange(len(error)), error)
    pylab.title('Relative Error for Each Sample on Test Data')
    pylab.ylabel('Error')
    pylab.xlabel('Sample Number')
    pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/error_per_sample.pdf'), bbox_inches='tight')'''



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
plt.show()

pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/data_decoded.pdf'), bbox_inches='tight')

# Plot alpha diversity before and after encoding

# plot beta diveristy before and after encoding


