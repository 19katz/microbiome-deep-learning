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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

backend = K.backend()

import load_kmer_cnts_jf
import deep_learning_models

#################
# Load the data # 
#################

kmer_size=6

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

y_pred=model.predict(data_normalized)
fpr, tpr, thresholds = roc_curve(labels, y_pred)
y_pred = (y_pred > 0.5)
conf_mat=confusion_matrix(labels, y_pred)
#auc= auc(fpr,tpr)
acc=accuracy_score(labels, y_pred)

# history is a dictionary. To get the keys, type print(history.history.keys())

#############
# plot roc: #
############# 
graph_dir = '~/deep_learning_microbiome/analysis/'
file_name=os.path.expanduser(graph_dir + 'roc.pdf')
deep_learning_models.plot_roc_aucs(fpr, tpr, auc, acc, file_name)

##########################
# Plot accuracy vs epoch #
##########################
graph_dir = '~/deep_learning_microbiome/analysis/'

pylab.figure()
pylab.plot(history.history['acc'])
pylab.plot(history.history['val_acc'])
pylab.legend(['training','test'], loc='upper right')

pylab.title('Model accuracy by epochs')
pylab.ylabel('Accuracy')
pylab.xlabel('Epoch')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .24, 'Number of encoding dimensions: {}'.format(encoding_dim))
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/accuracy.pdf') , bbox_inches='tight')



###########################
# Plot a confusion matrix #
###########################
file_name=os.path.expanduser(graph_dir + 'confusion_matrix.pdf')
classes=['0','1']
deep_learning_models.plot_confusion_matrix(conf_mat, classes,file_name)


######################################################
########################################################
# In this part of the code, I will vary different parameters to test how the predictions change
########################################################

############################################################################
# First test: how does number of encoding dimensions change the accuracy?  #
############################################################################  


encoding_dims=[1,2,10,50,100,200,300,400,500]

deep_learning_output={} #store the output in here. 

for encoding_dim in encoding_dims:
    deep_learning_output[encoding_dim]={'fpr':0,'tpr':0,'conf_mat':0, 'auc':0,'acc':0}

for encoding_dim in encoding_dims:
    input_dim=len(data_normalized[0]) # this is the number of input kmers
    #
    encoded_activation = 'relu'
    #encoded_activation = 'linear'
    #decoded_activation = 'softmax'
    decoded_activation = 'sigmoid'
    #
    loss='binary_crossentropy'
    #
    model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)
    #
    #weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'
    #
    #################
    # Fit the model #
    #################
    #
    numEpochs = 1000
    batchSize = 32
    #
    history = History()
    #
    model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])
    #
    y_pred=model.predict(data_normalized)
    fpr, tpr, thresholds = roc_curve(labels, y_pred)
    y_pred = (y_pred > 0.5)
    conf_mat=confusion_matrix(labels, y_pred)
    #auc= auc(fpr,tpr)
    acc=accuracy_score(labels, y_pred)
    #
    deep_learning_output[encoding_dim]['fpr']=fpr
    deep_learning_output[encoding_dim]['tpr']=tpr
    deep_learning_output[encoding_dim]['conf_mat']=conf_mat
    deep_learning_output[encoding_dim]['auc']=auc
    deep_learning_output[encoding_dim]['acc']=acc




# plot the roc curves for different numbers of encoding dimensions:

colors=['#543005','#8c510a','#bf812d','#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']
pylab.figure()

pylab.figure()
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
title='ROC as function of number of encoding dims'
pylab.title(title)

color_index=0
for encoding_dim in deep_learning_output:
    pylab.plot(deep_learning_output[encoding_dim]['fpr'], deep_learning_output[encoding_dim]['tpr'], color=colors[color_index])
    color_index +=1

pylab.legend(['1','2','10','50','100','200','300','400','500'], loc='upper right')
pylab.plot([0, 1], [0, 1], 'k--')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/ROC_vs_encoding_dims.pdf')
              , bbox_inches='tight')


# plot accuracy vs number of encoding dims
pylab.figure()
pylab.xlabel('Number of encoding dimensions')
pylab.ylabel('Accuracy')
title='Accuracy as function of number of encoding dims'
pylab.title(title)

accuracy_vector=[]
for encoding_dim in deep_learning_output:
    accuracy_vector.append(deep_learning_output[encoding_dim]['acc'])

pylab.plot(encoding_dims,accuracy_vector)

pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/Accuracy_vs_encoding_dims.pdf')
              , bbox_inches='tight')



for encoding_dim in deep_learning_output:
    print('\nencoding dim is %s' %encoding_dim)
    print(deep_learning_output[encoding_dim]['conf_mat'])


#################################
# second test: 5 mers vs 3 mers #
################################# 


# note im manually changing variables...
# 200 encoding dims

deep_learning_output={}
for mer in ['5mer','3mer']:
    deep_learning_output[mer]={'fpr':0,'tpr':0,'conf_mat':0, 'auc':0,'acc':0}

mer='3mer'
deep_learning_output[mer]['fpr']=fpr
deep_learning_output[mer]['tpr']=tpr
deep_learning_output[mer]['conf_mat']=conf_mat
deep_learning_output[mer]['acc']=acc



colors=['#8c510a', '#01665e']
pylab.figure()
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
title='ROC as function of kmer length'
pylab.title(title)

color_index=0
for kmer_len in ['3mer','5mer']:
    pylab.plot(deep_learning_output[kmer_len]['fpr'], deep_learning_output[kmer_len]['tpr'], color=colors[color_index])
    color_index +=1

pylab.legend(['3mer','5mer'], loc='upper right')

pylab.plot([0, 1], [0, 1], 'k--')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/ROC_vs_kmer_size.pdf')
              , bbox_inches='tight')




#################################
# One data set vs all data sets #
#################################
# note im manually changing variables...
# 200 encoding dims

deep_learning_output={}
for mer in ['all_datasets','1_dataset']:
    deep_learning_output[mer]={'fpr':0,'tpr':0,'conf_mat':0, 'auc':0,'acc':0}

mer='1_dataset'
deep_learning_output[mer]['fpr']=fpr
deep_learning_output[mer]['tpr']=tpr
deep_learning_output[mer]['conf_mat']=conf_mat
deep_learning_output[mer]['acc']=acc



colors=['#8c510a', '#01665e']
pylab.figure()
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
title='ROC as function of amount of data used for healthy control'
pylab.title(title)

color_index=0
for kmer_len in ['all_datasets','1_dataset']:
    pylab.plot(deep_learning_output[kmer_len]['fpr'], deep_learning_output[kmer_len]['tpr'], color=colors[color_index])
    color_index +=1

pylab.legend(['all datasets','1 dataset'], loc='upper right')

pylab.plot([0, 1], [0, 1], 'k--')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/ROC_vs_datasets.pdf')
              , bbox_inches='tight')




#################################
# type of encoding activation   #
#################################
deep_learning_output={}

encoded_activations=['relu', 'linear', 'softmax', 'tanh', 'elu',  'softplus', 'softsign', 'sigmoid', 'hard_sigmoid']

for encoded_activation in encoded_activations:
    deep_learning_output[encoded_activation]={'fpr':0,'tpr':0,'conf_mat':0, 'auc':0,'acc':0}

for encoded_activation in encoded_activations:

for encoded_activation in ['softplus', 'softsign', 'sigmoid', 'hard_sigmoid']:
    #
    input_dim=len(data_normalized[0]) # this is the number of input kmers
    encoding_dim=200
    #
    decoded_activation = 'sigmoid'
    #
    loss='binary_crossentropy'
    #
    model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation, decoded_activation)
    #
    #weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'
    #
    #################
    # Fit the model #
    #################
    #
    numEpochs = 1000
    batchSize = 32
    #
    history = History()
    #
    model.fit(data_normalized, labels, epochs=numEpochs, validation_split=0.2, batch_size=batchSize, shuffle=True, callbacks=[history])
    #model.fit(data_normalized, labels, epochs=numEpochs, batch_size=batchSize, shuffle=True, callbacks=[history])
    #
    y_pred=model.predict(data_normalized)
    fpr, tpr, thresholds = roc_curve(labels, y_pred)
    y_pred = (y_pred > 0.5)
    conf_mat=confusion_matrix(labels, y_pred)
    #auc= auc(fpr,tpr)
    acc=accuracy_score(labels, y_pred)
    #
    deep_learning_output[encoded_activation]['fpr']=fpr
    deep_learning_output[encoded_activation]['tpr']=tpr
    deep_learning_output[encoded_activation]['conf_mat']=conf_mat
    deep_learning_output[encoded_activation]['acc']=acc



# plot

#colors=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

pylab.figure()
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
title='ROC as function of encoding activation function'
pylab.title(title)

color_index=0
for encoded_activation in encoded_activations:
    pylab.plot(deep_learning_output[encoded_activation]['fpr'], deep_learning_output[encoded_activation]['tpr'], color=colors[color_index])
    color_index +=1

pylab.legend(encoded_activations, loc='upper right')
pylab.plot([0, 1], [0, 1], 'k--')


pylab.gca().set_position((.1, .6, .8, .6))
pylab.figtext(0.02, .4, 'This graph shows how loss changes with number of epochs for different splits between training and test data.')
pylab.figtext(0.02, .32, 'Backend: ' + backend)
pylab.figtext(0.02, .28, 'Loss function: ' + loss)
pylab.figtext(0.02, .16, 'Number of epochs of training: {}'.format(numEpochs))
pylab.figtext(0.02, .12, 'Batch size used during training: {}'.format(batchSize))
pylab.figtext(0.02, .08, 'Activation function used for encoding: ' + encoded_activation)
pylab.figtext(0.02, .04, 'Activation function used for decoding: ' + decoded_activation)
pylab.savefig(os.path.expanduser(graph_dir + '/ROC_vs_encoded_activations.pdf')
              , bbox_inches='tight')








###################################
# deep learning vs Random Forest  #
###################################


#################################################
# With and without pre-training for autoencoder #
#################################################

