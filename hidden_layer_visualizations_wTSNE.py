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

kmer_size=7

#data_sets_healthy=['HMP', 'Qin_et_al','RA','MetaHIT','Feng','Karlsson_2013','LiverCirrhosis','Zeller_2014']

data_sets_healthy=['MetaHIT']
allowed_labels=['0']
kmer_cnts_healthy, accessions_healthy, labels_healthy, domain_labels =load_kmer_cnts_jf.load_kmers(kmer_size,data_sets_healthy, allowed_labels)

data_sets_diseased=['MetaHIT']
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

######################################                                                                                                                     
# TsNE before putting into the model #                                                                                                                     
######################################

X = data_normalized
y = np.array(labels)

X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
plt.figure()
y_test_cat = np_utils.to_categorical(y, num_classes = 2)
color_map = np.argmax(y_test_cat, axis=1)
for cl in range(2):
    indices = np.where(color_map==cl)
    indices = indices[0]
    plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
plt.legend(('Healthy', 'Diseased'))
plt.title('TsNE Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_before_model.pdf")

###############################################
# standalone model without varying dimensions #
###############################################

#input_dim=len(data_normalized[0]) # this is the number of input kmers
#encoding_dim=8

#encoded_activation = 'relu'
#encoded_activation = 'sigmoid'
#encoded_activation = 'linear'
#decoded_activation = 'softmax'
#decoded_activation = 'sigmoid'
#decoded_activation = 'softmax'

#loss='binary_crossentropy'

#model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)

#numEpochs = 1000
#batchSize = 32

#history = History()

#model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])

###############################                                                                                                                                 
# get encoded weights to TsNE #                                                                                                                                 
###############################

#def create_truncated_model(trained_model): 
#    model = Sequential() 
#    model.add(Dense(encoding_dim, activation=encoded_activation, input_dim=input_dim))
#    for i, layer in enumerate(model.layers): 
#        layer.set_weights(trained_model.layers[i].get_weights())
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
#    return model

#truncated_model = create_truncated_model(model)
#hidden_features = truncated_model.predict(data_normalized)

# Set up X and y for sklearn and numpy                                                                                                                        
#X = hidden_features
#y = np.array(labels)

#TsNE plot                                                                                                                                                     
#X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
#plt.figure()
#y_test_cat = np_utils.to_categorical(y, num_classes = 2)
#color_map = np.argmax(y_test_cat, axis=1)
#for cl in range(2):
#    indices = np.where(color_map==cl)
#    indices = indices[0]
#    plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
#plt.legend(('Healthy', 'Diseased'))
#plt.title('TsNE Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_encoded_weights.pdf")


#############################################
# Visualize hidden layer activations w/TsNE #
#############################################

#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.layers[0].output)
#intermediate_output = intermediate_layer_model.predict(data_normalized)

#X = intermediate_output
#y = np.array(labels)

#X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)                                                      
#plt.figure()
#y_test_cat = np_utils.to_categorical(y, num_classes = 2)
#color_map = np.argmax(y_test_cat, axis=1)
#for cl in range(2):
#    indices = np.where(color_map==cl)
#    indices = indices[0]
#    plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
#plt.legend(('Healthy', 'Diseased'))
#plt.title('TsNE Plot hidden layer output' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')                                                         
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" +str(kmer_size) + "mers/TsNE_hiddenlayer_output.pdf") 

######################################                                                                                                                          
# Visualize full model output w/TsNE #                                                                                                                          
######################################

#final_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
#final_output = final_layer_model.predict(data_normalized)

#X = final_output
#y = np.array(labels)

#TsNE plot                                                                                                                                                                                                #X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
#plt.figure()
#y_test_cat = np_utils.to_categorical(y, num_classes = 2)
#color_map = np.argmax(y_test_cat, axis=1)
#for cl in range(2):
#    indices = np.where(color_map==cl)
#    indices = indices[0]
#    plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
#plt.legend(('Healthy', 'Diseased'))
#plt.title('TsNE Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_full_model_output.pdf")

#######################################################################                                                                                                                               
# How does changing t-SNE parameters change the output visualization? #                                                                                                                                
#######################################################################

# Function
def draw_tsne(n_components=2, perplexity=30, learning_rate=200, title=''):
    fit = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
    )
    u = fit.fit_transform(X);
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title(title, fontsize=18)
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/" + title + ".pdf")

############################################################################                                                                                                                               
# How does number of encoding dimensions change the output visualization? #                                                                                                                               
############################################################################                                                                                                                               

# t-SNE
 
encoding_dims = []
#encoding_dims=[8,300,4000]                                                                                                                                                              

for encoding_dim in encoding_dims:
    input_dim=len(data_normalized[0]) # this is the number of input kmers                                                                                                                                 

    encoded_activation = 'relu'
    #encoded_activation = 'sigmoid'
    #encoded_activation = 'linear'                                                                                                                                                                        
    #decoded_activation = 'softmax'                                                                                                                                                                       
    decoded_activation = 'sigmoid'

    loss='binary_crossentropy'

    model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)

    # Fit the model #                                                                                                                                                                                     
    numEpochs = 1000
    batchSize = 32
    history = History()
    model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])

    #final_output = model.predict(data_normalized)
    final_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
    final_output = final_layer_model.predict(data_normalized)

    X = final_output
    y = np.array(labels)

    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('TsNE Plot final layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_finallayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")


# t-SNE tuning

#encoding_dims = []
encoding_dims=[8,300,4000]                                                                                                                                                                                 

for encoding_dim in encoding_dims:
    input_dim=len(data_normalized[0]) # this is the number of input kmers                                                                                                                                  

    encoded_activation = 'relu'
    #encoded_activation = 'sigmoid'                                                                                                                                                                        
    #encoded_activation = 'linear'                                                                                                                                                                         
    #decoded_activation = 'softmax'                                                                                                                                                                        
    decoded_activation = 'sigmoid'

    loss='binary_crossentropy'

    model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)

    numEpochs = 1000
    batchSize = 32
    history = History()
    model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])

    #final_output = model.predict(data_normalized)                                                                                                                                                         
    final_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
    final_output = final_layer_model.predict(data_normalized)

    X = final_output
    y = np.array(labels)

    #perplexity                                                                                                                                                                                            
    for p in (2, 5, 10, 20, 30, 40, 50):
        draw_tsne(perplexity=p, title=str(encoding_dim) + 'perplexity = {}'.format(p))

    #learning_rate                                                                                                                                                                                         
    for l in (10, 50, 100, 200, 500):
        draw_tsne(learning_rate=l, title=str(encoding_dim) + 'learning_rate = {}'.format(l))

