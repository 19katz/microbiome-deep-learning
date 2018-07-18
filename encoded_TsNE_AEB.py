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
from keras.layers.core import Dropout
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
import itertools
from itertools import cycle, product
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

backend = K.backend()

import load_kmer_cnts_jf
import deep_learning_models

#################
# Load the data # 
#################

kmer_size=5

#data_sets_healthy=['HMP', 'Qin_et_al','RA','MetaHIT','Feng','Karlsson_2013','LiverCirrhosis','Zeller_2014']

data_sets_healthy=['Qin_et_al']
allowed_labels=['0']
kmer_cnts_healthy, accessions_healthy, labels_healthy, domain_labels =load_kmer_cnts_jf.load_kmers(kmer_size,data_sets_healthy, allowed_labels)

data_sets_diseased=['Qin_et_al']
allowed_labels=['1']
kmer_cnts_diseased, accessions_diseased, labels_diseased, domain_labels =load_kmer_cnts_jf.load_kmers(kmer_size,data_sets_diseased, allowed_labels)

kmer_cnts=np.concatenate((kmer_cnts_healthy,kmer_cnts_diseased))
accessions=np.concatenate((accessions_healthy,accessions_diseased))
labels=np.concatenate((labels_healthy,labels_diseased))

labels=np.asarray(labels)
labels=labels.astype(np.int)
healthy=np.where(labels==0)
disease=np.where(labels==1)


data=pd.DataFrame(kmer_cnts)
data_normalized = normalize(data, axis = 1, norm = 'l1')

data_normalized, labels = shuffle(data_normalized, labels, random_state=0)

###########################################
# set up a model (supervised learning)    #
###########################################

input_dim=len(data_normalized[0]) # this is the number of input kmers
encoding_dim=200

encoded_activation = 'relu'
#encoded_activation = 'linear'
#decoded_activation = 'softmax'
decoded_activation = 'sigmoid'

loss='binary_crossentropy'

model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)

#weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'

#################
# Fit the model #
#################

numEpochs = 1000
batchSize = 32

history = History()

model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])
#model.fit(data_normalized, labels, epochs=numEpochs, batch_size=batchSize, shuffle=True, callbacks=[history])

graph_dir = '~/deep_learning_microbiome/analysis/'

###################################                                                                                                                                                                
# get encoded information to TsNE #                                                                                                                                                                
###################################

def create_truncated_model(trained_model): 
    model = Sequential() 
    model.add(Dense(encoding_dim, activation=encoded_activation, input_dim=input_dim))
    for i, layer in enumerate(model.layers): 
        layer.set_weights(trained_model.layers[i].get_weights())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

truncated_model = create_truncated_model(model)
hidden_features = truncated_model.predict(data_normalized)

###########################################                                                                                                 
# TsNE Visualization of encoded dimension #                                                                                                                                                                
###########################################

# Set up X and y for sklearn and numpy                                                                                                                                                                    
X = hidden_features
y = np.array(labels)

#TsNE plot                                                                                                                                                                                                 
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X.data)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Spectral", alpha = 0.7)
plt.title('TsNE Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/TsNE_encoded" + str(data_sets_healthy) + str(kmer_size) + ".pdf")

############################################################################
# First test: how does number of encoding dimensions change the accuracy?  #
############################################################################  
