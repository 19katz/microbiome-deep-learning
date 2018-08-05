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
import pickle
from importlib import reload
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

import load_kmer_cnts_jf


#################                                                                                                                            
# Load the data #                                                                                                                            
#################                                                                                                                            

#This is temporary code I used so that I could pickle loaded kmer sets and take them locally to work on QC_plots.py.

kmer_size=10 #change depending on datset to be loaded

#data_sets_healthy=['HMP', 'Qin_et_al', 'RA', 'MetaHIT', 'Feng', 'Karlsson_2013', 'LiverCirrhosis', 'Zeller_2014']
data_sets_healthy= ['RA']
num_data_sets=len(data_sets_healthy)

allowed_labels=['0', '1'] #change depending on whether you want all labels

kmer_cnts_healthy, accessions_healthy, labels_healthy, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size,data_sets_healthy, allowed_labels)

kmer_cnts=kmer_cnts_healthy
accessions=accessions_healthy
labels=labels_healthy
domains = domain_labels

labels=np.asarray(labels)
labels=labels.astype(np.int)

data=pd.DataFrame(kmer_cnts)
#data_normalized = normalize(data, axis = 1, norm = 'l1')
#accessions = pd.DataFrame(accessions)
#accessions.to_csv("/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/karlsson10mers.csv")
data.to_pickle("/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/RA10mers.pickle")
data_labels = pd.DataFrame(labels)
data_labels.to_csv("/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/RA10mers.csv", header=False, index=False)


