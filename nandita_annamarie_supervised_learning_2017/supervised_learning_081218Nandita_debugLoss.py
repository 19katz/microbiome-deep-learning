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
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from scipy import interp
import argparse

backend = K.backend()

# import our private scripts
import load_kmer_cnts_jf
import deep_learning_models
import plotting_utils
import stats_utils
import config_file_local as config_file
import importlib
import imp
import os

# read in directories:
data_directory = config_file.data_directory
analysis_directory = config_file.analysis_directory
scripts_directory = config_file.scripts_directory


for kmer_size in [5]:
    #
    #################
    # Load the data # 
    #################
    print('Loading data...')
    #
    data_set='Qin_et_al'
    kmer_size=5
    n_splits=5
    n_repeats=1
    encoding_dim=10
    dropout_pct=0.5
    input_dropout_pct=0
    encoded_activation='relu'
    num_epochs=400
    #
    data_normalized, kmer_cnts, labels, rskf = load_kmer_cnts_jf.load_single_disease(data_set, kmer_size, n_splits, n_repeats, precomputed_kfolds=False, bootstrap = True)
    #
    bootstrapped_data=stats_utils.bootstrap_data(data_normalized, kmer_cnts, 100, 100000)
    #
    ###################################################
    # iterate through the data kfolds and iterations #
    ###################################################
    #
    for n_repeat in range(0,1):
        #
        train_idx = rskf[0][n_repeat]
        test_idx = rskf[1][n_repeat]
        
        #access the bootstrapped data
        bootstrapped_data_stacked=np.asarray(bootstrapped_data[train_idx[0]])
        for idx in train_idx[1:]:
            bootstrapped_data_stacked=np.vstack((bootstrapped_data_stacked,np.asarray(bootstrapped_data[idx])))
    
        bootstrapped_data_normalized = normalize(bootstrapped_data_stacked, axis = 1, norm = 'l1')

        #add on the real training data too
        bootstrapped_data_normalized=np.vstack((bootstrapped_data_normalized, data_normalized[train_idx]))
        
        training_labels=[]
        for idx in train_idx:
            for i in range(0, num_replicates):
                training_labels.append(labels[idx])
                
        for idx in train_idx:
            training_labels.append(labels[idx])

        x_train, y_train = bootstrapped_data_normalized, np.asarray(training_labels)
        x_test, y_test = data_normalized[test_idx], labels[test_idx]
        #
        #standardize the data, mean=0, std=1
        norm_input=True
        if norm_input:
            x_train, x_test= stats_utils.standardize_data_bootstrap(data_normalized[train_idx], x_test, x_train)
        #
        ###########################################
        # set up a model (supervised learning)    #
        ###########################################
        # note that the model has to be instantiated each time a new fold is started otherwise the weights will not start from scratch. 
        #
        input_dim=len(data_normalized[0]) # this is the number of input kmers
        #
        model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation,input_dropout_pct, dropout_pct)
        #
        ##################################################
        # Fit the model with the train data of this fold #
        ##################################################
        history = History()
        # history is a dictionary. To get the keys, type print(history.history.keys())
        #
        model.fit(x_train, y_train, 
                  epochs=num_epochs, 
                  batch_size=len(x_train), 
                  shuffle=True,
                  validation_data=(x_test, y_test),
                  verbose=1,
                  callbacks=[history])
        #
        plotting_str='bootstrapped_qin_10x_relu_dropout75'
        # plot loss vs epoch
        plotting_utils.plot_loss_vs_epoch(history, analysis_directory, plotting_str=plotting_str)
        plotting_utils.plot_accuracy_vs_epoch(history, analysis_directory, plotting_str=plotting_str)
        
        

