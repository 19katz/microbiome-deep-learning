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
from sklearn.decomposition import PCA

import umap

backend = K.backend()

import load_kmer_cnts_jf
import deep_learning_models

#############
# Functions #
#############

# plot tSNE                                                                                                                                                                                                
def plot_TSNE(X, y, layer = ''): # set layer to hidden or final
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)                                                                                                            
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('tSNE' + layer + 'layer output')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/kmers/tSNEoutput_" + layer + ".pdf")

# plot UMAP
def plot_UMAP(X, y, layer = ''):
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('UMAP' + layer + 'layer output')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/kmers/UMAPoutput_" + layer + ".pdf")

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

#normalizing across rows
data_normalized = normalize(data, axis = 1, norm = 'l1')

#normalizing within columns
#sample_mean = data_normalized.mean(axis=0)
#sample_std = data_normalized.std(axis=0)

# Normalize both training and test samples with the training mean and std
#data_normalized = (data_normalized - sample_mean) / sample_std

data_normalized, labels = shuffle(data_normalized, labels, random_state=0)

######################################                                                                       
# TsNE/UMAP before putting into the model #                                                                       
######################################

X = data_normalized
y = np.array(labels)

plot_TSNE(X, y, layer = 'none')
plot_UMAP(X, y, layer = 'none')


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
#plot_TSNE(X, y, layer = 'hidden_weights')


#############################################
# Visualize hidden layer activations w/TsNE #
#############################################

#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.layers[0].output)
#intermediate_output = intermediate_layer_model.predict(data_normalized)

#X = intermediate_output
#y = np.array(labels)

#plot_TSNE(X, y, layer = 'hidden_activations')


######################################                                                                                                                          
# Visualize full model output w/TsNE #                                                                                                                          
######################################

#final_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
#final_output = final_layer_model.predict(data_normalized)

#X = final_output
#y = np.array(labels)

#TsNE plot                                                                                                                                                                                                #
#plot_TSNE(X, y, layer = 'final')


#######################################################################
# How does changing t-SNE parameters change the output visualization? #
#######################################################################

# Notes
#perplexity should be between 5 and 50; not super critical
#learning_rate should be between 10 and 1000
#n_ter should be at least 250
#init can be 'pca' or 'random'

# Function
def draw_tsne(n_components=2, perplexity=30, learning_rate=200, init='random', n_iter=1000, title=''):
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

# t-SNE tuning                                                                                                                                                   
encoding_dims = []
#encoding_dims=[300]                                                                                                                                                                                     
for encoding_dim in encoding_dims:
    input_dim=len(data_normalized[0]) # this is the number of input kmers                                                                                                                                 
                                                                                                                                                                                                          
    encoded_activation = 'relu'
    decoded_activation = 'sigmoid'

    loss='binary_crossentropy'

    model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)

    numEpochs = 1000
    batchSize = 32
    history = History()
    model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])

    final_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
    final_output = final_layer_model.predict(data_normalized)

    X = final_output
    y = np.array(labels)

    #perplexity                                                                                                                                                                                           
    for p in (10,30,50,100):
        draw_tsne(perplexity=p, title=str(encoding_dim) + 'perplexity = {}'.format(p))

    #learning_rate                                                                                                                                                                                        
    for l in (10, 100, 500, 1000):
        draw_tsne(learning_rate=l, title=str(encoding_dim) + 'learning_rate = {}'.format(l))

    #n_iter                                                                                                                                                                                               
    for n in (250, 1000, 5000):
        draw_tsne(n_iter=n, title=str(encoding_dim) + 'n_iter = {}'.format(n))

    #init                                                                                                                                                                                                 
    #draw_tsne(init='pca', title=str(encoding_dim) + 'pca_init') 

############################################################################                                                                                                                              # How does number of encoding dimensions change the output visualization? #                                                                                                                               
############################################################################                                                                                                                              

#final output not compatible with PCA for some reason

#encoding_dims=[int(len(data_normalized[0])*1/4), int(len(data_normalized[0])*1/2), int(len(data_normalized[0])*3/4)] 
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

    #TsNE                                                                                                                                                                                                 
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('TsNE final layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_finallayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #UMAP                                                                                                                                                                                                 
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('UMAP Plot final layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/UMAP_finallayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

####################################################################
# Visualize hidden layer activations with varied encoding dimensions #
####################################################################

encoding_dims = []
#encoding_dims=[int(len(data_normalized[0])*1/4), int(len(data_normalized[0])*1/2), int(len(data_normalized[0])*3/4)]
#encoding_dims = [int(len(data_normalized[0])*3/4)]
#encoding_dims = [100, 200, 300, 400, 500]

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

#    penult_out_model = Model(inputs=model.input, outputs=model.layers[0].output)
    penult_out_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
#    penult_out_model = Model(inputs=model.layers[0].input, outputs=model.layers[(len(model.layers) - 1)].output)
    penult_out = penult_out_model.predict(data_normalized)

    X = penult_out
    y = np.array(labels)

    #PCA
    X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_pca[indices,0], X_pca[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('PCA hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/PCA_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")
    
    #TsNE
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('TsNE hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #PCA2TsNE
    X_pca = PCA(n_components=25, random_state=0).fit_transform(X)
    X_tsne_pca = TSNE(n_components=2, random_state=0).fit_transform(X_pca.data)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne_pca[indices,0], X_tsne_pca[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('PCA2TsNE hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/PCA2TsNE_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #UMAP
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('UMAP Plot hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/UMAP_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")


#############################
# hidden and final together #
#############################

#encoding_dims=[int(len(data_normalized[0])*1/4), int(len(data_normalized[0])*1/2), int(len(data_normalized[0])*3/4)]                                                               
encoding_dims = []

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

    final_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
    final_output = final_layer_model.predict(data_normalized)

    penult_out_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
    penult_out = penult_out_model.predict(data_normalized)

    X1 = final_output
    X2 = penult_out
    y = np.array(labels)

    #Final visualizations
    #TsNE                                                                                                                                                                                                 
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X1)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('TsNE final layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_finallayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #UMAP                                                                                                                                                                                                 
    fit = umap.UMAP()
    u = fit.fit_transform(X1)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('UMAP Plot final layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/UMAP_finallayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #Hidden visualizations
    #PCA                                                                                                                                                                                                  
    X_pca = PCA(n_components=2, random_state=0).fit_transform(X2)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_pca[indices,0], X_pca[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('PCA hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/PCA_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #TsNE                                                                                                                                                                                                 
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X2)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne[indices,0], X_tsne[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('TsNE hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/TsNE_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

    #PCA2TsNE                                                                                                                                                                                             
    X_pca = PCA(n_components=25, random_state=0).fit_transform(X2)
    X_tsne_pca = TSNE(n_components=2, random_state=0).fit_transform(X_pca.data)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(X_tsne_pca[indices,0], X_tsne_pca[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('PCA2TsNE hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/PCA2TsNE_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pd\
f")

    #UMAP                                                                                                                                                                                                 
    fit = umap.UMAP()
    u = fit.fit_transform(X2)
    plt.figure()
    y_test_cat = np_utils.to_categorical(y, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(u[indices,0], u[indices, 1], label=cl, alpha = 0.7)
    plt.legend(('Healthy', 'Diseased'))
    plt.title('UMAP Plot hidden layer output, dims = ' + str(encoding_dim) + ", " + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/" + str(kmer_size) + "mers/UMAP_hiddenlayeroutput_dims" + str(encoding_dim) + str(data_sets_healthy) + str(kmer_size) + ".pdf")

