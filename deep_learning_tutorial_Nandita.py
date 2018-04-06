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
from importlib import reload
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K

backend = K.backend()

import load_kmer_cnts_jf
import deep_learning_models

#################
# Load the data # 
#################

#kmer_size=5
kmer_size=10
data_set='Qin_et_al'
#data_set='RA'
#data_set='MetaHIT'
#data_set='HMP'

kmer_cnts, accessions, labels =load_kmer_cnts_jf.load_kmers(kmer_size,data_set)
labels=np.asarray(labels)
healthy=np.where(labels=='0')
disease=np.where(labels=='1')


data=pd.DataFrame(kmer_cnts)
data_normalized = normalize(data, axis = 1, norm = 'l1')


################################
# set up a model (autoencoder)
################################

input_dim=len(data_normalized[0]) # this is the number of input kmers
encoding_dim=10000

encoded_activation = 'relu'
#encoded_activation = 'linear'
decoded_activation = 'softmax'
#decoded_activation = 'sigmoid'
# sigmoid used to give values between 0 and 1
loss='kullback_leibler_divergence'
bias=True
autoencoder=deep_learning_models.create_autoencoder(encoding_dim, input_dim, encoded_activation, decoded_activation)

#weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'

#################
# Fit the model #
#################

cuts= np.array([50,100,150,200,250])
#tests = np.subtract(1573, cuts)
allHistories = []
numEpochs = 1000
batchSize = 25
#np.random.shuffle(data_normalized)
for cut in cuts:
    x_train=data_normalized[:cut]
    x_test=data_normalized[cut:]
    history = History()
    #autoencoder.load_weights(weightFile)
    autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[history])
    allHistories.append(history)

# history is a dictionary. To get the keys, type print(history.history.keys())


###########
# Plot!   #
###########

graph_dir = '~/deep_learning_microbiome/analysis/'
# file naming system
fileInfo = '{}{}'.format('_encDim-', encoding_dim) + '{}{}'.format('_bias-', bias) + '{}{}'.format('_numEpochs-', numEpochs) + '{}{}'.format('_batch-',  batchSize) + '_encActFunc-' + encoded_activation + '_decActFunc-' + decoded_activation + '_lossFunc-' + loss + '_backend-' + backend


################################################################
# how does the loss change as the fraction of data in the test vs training change?
############################################################

colors=['#addd8e','#fecc5c','#fd8d3c','#f03b20','#bd0026']

pylab.figure()

color_index=0
for hs in allHistories:
    pylab.plot(hs.history['val_loss'], color=colors[color_index])
    color_index +=1

pylab.legend(['50', '100','150','200','250'], loc='upper right')


color_index=0
for hs in allHistories:
    pylab.plot(hs.history['loss'],'--', color=colors[color_index])
    color_index +=1

pylab.title('Model loss by epochs and number of training samples')
pylab.ylabel('Test Loss (KL divergence)')
pylab.xlabel('Epoch')


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





######################################################
# how do the points before and after encoding compare (can we make a y=x line?)
######################################################


cut = 250
x_train=data_normalized[:cut]
x_test=data_normalized[cut:]

#x_train=data_normalized
#x_test=data_normalized
history = History()
callbacks = [history]
if backend == 'tensorflow':
    callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))

numEpochs=100
#autoencoder.load_weights(weightFile)
autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=callbacks)

# Make new predictions #
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

testTrainInfo ='{}{}'.format('_numTrain-', x_train.shape[0]) + '{}{}'.format('_numTest-', x_test.shape[0])

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

# plot y=x line
#lims = [
#    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#]

lims = [min(min(input_data), min(decoded_data)),  max(max(input_data), max(decoded_data)) ]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)



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


###############################################################
# The plot!!! Number of encodign dimensions vs KL divergence  #
###############################################################

################################
# set up a model (autoencoder)
################################

#encoding_dims=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,512]

# for 10mers:
encoding_dims=[5,50,100,1000,10000]

encoded_activation = 'relu'
#encoded_activation = 'linear'
decoded_activation = 'softmax'
#decoded_activation = 'sigmoid'
# sigmoid used to give values between 0 and 1

numEpochs = 1000
batchSize = 100

allHistories = []

for encoding_dim in encoding_dims:
    print(encoding_dim)
    autoencoder=deep_learning_models.create_autoencoder(encoding_dim, input_dim, encoded_activation, decoded_activation)
    #
    cut = 200
    x_train=data_normalized[:cut]
    x_test=data_normalized[cut:]
    #
    history = History()
    callbacks = [history]
    if backend == 'tensorflow':
        callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))
    #    
    numEpochs=100
    #autoencoder.load_weights(weightFile)
    autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=callbacks)
    allHistories.append(history)




graph_dir = '~/deep_learning_microbiome/analysis/'

pylab.figure()

color_index=0

from matplotlib.pyplot import cm 
color=iter(cm.rainbow(np.linspace(0,1,len(encoding_dims))))

for hs in allHistories:
    c=next(color)
    pylab.plot(hs.history['val_loss'], c=c)

#for 5mers:
#pylab.legend(['1','2','3','4','5','6','7','8','9','10','15','20','25','30','40','50','60','70','80','90','100','125','150','175','200','250','300','350','400','450','500','512'], loc='upper right')

#for 10mers:

pylab.legend=(['5','50','100','1000','10000'],loc='upper right')

pylab.title('Model loss by epochs and number of encoding dimensions')
pylab.ylabel('Test Loss (KL divergence)')
pylab.xlabel('Epoch')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different numbers of encoding dimensions.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
#pylab.figtext(0.02, .24, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .2, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)



pylab.savefig(os.path.expanduser(graph_dir + '/number_inner_nodes_vs_loss'
                                 + '.pdf')
              , bbox_inches='tight')



# Go through and store epoch 40 in a vector
epoch_40_ReLU =[]
for hs in allHistories:
    epoch_40_ReLU.append(hs.history['val_loss'][40])


epoch_80_ReLU =[]
for hs in allHistories:
    epoch_80_ReLU.append(hs.history['val_loss'][80])

epoch_20_ReLU =[]
for hs in allHistories:
    epoch_20_ReLU.append(hs.history['val_loss'][20])



# repeat for linear encoding:

#encoded_activation = 'relu'
encoded_activation = 'linear'
decoded_activation = 'softmax'
#decoded_activation = 'sigmoid'
# sigmoid used to give values between 0 and 1

numEpochs = 1000
batchSize = 100

allHistories_linear = []

for encoding_dim in encoding_dims:
    autoencoder=deep_learning_models.create_autoencoder(encoding_dim, input_dim, encoded_activation, decoded_activation)
    #
    cut = 200
    x_train=data_normalized[:cut]
    x_test=data_normalized[cut:]
    #
    history = History()
    callbacks = [history]
    if backend == 'tensorflow':
        callbacks.append(TensorBoard(log_dir='/tmp/autoencoder'))
    #    
    numEpochs=100
    #autoencoder.load_weights(weightFile)
    autoencoder.fit(x_train, x_train,
                    epochs=numEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=callbacks)
    allHistories_linear.append(history)





pylab.figure()

color_index=0

from matplotlib.pyplot import cm 
color=iter(cm.rainbow(np.linspace(0,1,len(encoding_dims))))

for hs in allHistories_linear:
    c=next(color)
    pylab.plot(hs.history['val_loss'], c=c)

#for 5mers:
#pylab.legend(['1','2','3','4','5','6','7','8','9','10','15','20','25','30','40','50','60','70','80','90','100','125','150','175','200','250','300','350','400','450','500','512'], loc='upper right')

#for 10mers:
pylab.legend=(['5','50','100','1000','10000'], loc='upper right')

pylab.title('Model loss by epochs and number of encoding dimensions')
pylab.ylabel('Test Loss (KL divergence)')
pylab.xlabel('Epoch')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different numbers of encoding dimensions.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
#pylab.figtext(0.02, .24, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .2, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)



pylab.savefig(os.path.expanduser(graph_dir + '/number_inner_nodes_vs_loss_linear'
                                 + '.pdf')
              , bbox_inches='tight')




# Go through and store epoch 40 in a vector
epoch_40_linear =[]
for hs in allHistories_linear:
    epoch_40_linear.append(hs.history['val_loss'][40])

epoch_80_linear =[]
for hs in allHistories_linear:
    epoch_80_linear.append(hs.history['val_loss'][80])

epoch_20_linear =[]
for hs in allHistories_linear:
    epoch_20_linear.append(hs.history['val_loss'][20])


# Plot epoch 40 KL loss for linear vs ReLU


pylab.figure()

pylab.plot(encoding_dims, epoch_80_ReLU, c='r')
pylab.plot(encoding_dims, epoch_80_linear, c='b')

pylab.legend(['ReLU','linear'], loc='upper right')

pylab.title('Model loss by and number of encoding dimensions at Epoch 80')
pylab.ylabel('Test Loss (KL divergence)')
pylab.xlabel('Number of encoding dimensions')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes for different numbers of encoding dimensions.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
#pylab.figtext(0.02, .24, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .2, 'Use bias: {}'.format(bias))
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
#pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)



pylab.savefig(os.path.expanduser(graph_dir + '/number_inner_nodes_vs_loss_linear_ReLU_epoch80'
                                 + '.pdf')
              , bbox_inches='tight')
