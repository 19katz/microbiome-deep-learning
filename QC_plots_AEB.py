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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

import load_kmer_cnts_jf


#################                                                                                                                                                 
# Load the data #                                                                                                                                                 
#################                                                                                                                                                  

kmer_size=5 #change depending on datset to be loaded                                                                                                               

#3mers and 5mers
#data_sets_healthy=['Qin_et_al', 'RA', 'MetaHIT', 'Feng', 'Karlsson_2013', 'LiverCirrhosis', 'Zeller_2014']                                                  

#10mers
#data_sets_healthy=['Qin_et_al', 'RA', 'MetaHIT']

#For one at a time
data_sets_healthy= ['Zeller_2014']

allowed_labels=['0', '1'] #change depending on whether you want all labels ['0','1'], or just one
kmer_cnts_healthy, accessions_healthy, labels_healthy, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size, data_sets_healthy, allowed_labels)
kmer_cnts=kmer_cnts_healthy
accessions=accessions_healthy
labels=labels_healthy
domains = domain_labels

labels=np.asarray(labels)
labels=labels.astype(np.int)
dataset_labels = labels
    
#Getting things into dataframe for ease
data=pd.DataFrame(kmer_cnts)
norm_array = normalize(data, axis = 1, norm = 'l1')                                                        
df_norm = pd.DataFrame(norm_array)
df_norm['Dataset'] = dataset_labels
    
# Set up X and y for sklearn and numpy
X = norm_array
y = np.array(dataset_labels)

V = X.T
    
#NMF Reconstruction error
plt.figure()
plt.xlabel('Number of Kmer signatures')
plt.ylabel('Frobenius reconstruction error')
no_components = 512
for i in range(1, no_components + 1):
    model = NMF(n_components = i, init='random', beta_loss = 'itakura-saito', 
                solver = 'mu', random_state=0, max_iter = 1000)
    W = model.fit_transform(V)
    recon_err = model.reconstruction_err_
    plt.scatter(i, recon_err, color = 'b')
plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/NMF.pdf")

#PCA plot
#X_pca = PCA(n_components=2, random_state=0).fit_transform(X.data)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Spectral",alpha = 0.7)
#plt.title('PCA Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
#plt.colorbar()
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/PCA" + str(data_sets_healthy) + str(kmer_size) + ".png")
    
#TsNE plot
#X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X.data)
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Spectral", alpha = 0.7)
#plt.title('TsNE Plot for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/TsNE" + str(data_sets_healthy) + str(kmer_size) + ".png")
    
#PCA into TsNE
#X_new_pca = PCA(n_components=25, random_state=0).fit_transform(X.data)
#X_pca_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_new_pca.data)
#plt.scatter(X_pca_tsne[:, 0], X_pca_tsne[:, 1], c=y, cmap="Spectral", alpha = 0.7)
#plt.title('PCA&TsNE for ' + str(data_sets_healthy) + ' ' + str(kmer_size) + 'mers')
#plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/PCA2TsNE" + str(data_sets_healthy) + str(kmer_size) + ".png")
