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
import pickle

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
tmp_intermediate_directory=config_file.tmp_intermediate_directory

for kmer_size in [5,6,7,8,10]:
    
    print(kmer_size)
    #################
    # Load the data # 
    #################
    print('Loading data...')
    
    data_set='Qin_et_al'

    data_normalized, kmer_cnts, labels, rskf = load_kmer_cnts_jf.load_single_disease(data_set, kmer_size, n_splits, n_repeats, precomputed_kfolds=False, bootstrap = True)
    
    num_replicates=100
    num_kmers=100000
    bootstrapped_data=stats_utils.bootstrap_data(data_normalized, kmer_cnts, num_replicates, num_kmers)

    pickle.dump(bootstrapped_data, open( "%skmer_size_%s_Qin_bootstrap.p"  %(tmp_intermediate_directory,kmer_size), "wb" ) )
        
    
