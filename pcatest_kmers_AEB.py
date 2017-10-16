import csv
import bz2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import pylab
import os
import pickle
#from loadkmers import sparse_kmers_df



################
# reading in data with for loop
################

#'''
#col = 0
#firstline = True
#with open("relative_abundance.txt") as tsv:
#    for line in csv.reader(tsv, dialect="excel-tab"):
#        if not firstline:
#            data[:, col] = line[1:]
#            col += 1
#        firstline = False
#'''

################
# read in data with pandas
################

#print(sparse_kmers_df.values.shape)

#Pickle time
data = pd.read_pickle('/pollard/home/abustion/play/pickles/fullfinalstack.pickle')

pca = PCA(n_components=2)
pca.fit(data)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_new = pca.transform(data)
print(data_new)
print("Shape")
print(data_new.shape)

pylab.figure()

pylab.title('PC2 VS PC1')

pylab.xlabel('PC1')
pylab.ylabel('PC2')

graph_dir = '/pollard/home/abustion/play/analysis'

pylab.scatter(data_new[:, 0],data_new[:, 1])
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph shows the relative abundance data plotted with 2-component PCA')
pylab.savefig(os.path.expanduser(graph_dir + '/pca_two_components' + '.pdf'), bbox_inches='tight')



pylab.figure()
pylab.title('Explained Variance vs Number of Components')
pylab.xlabel('Components')
pylab.ylabel('Total Variance')

nComponents = 10
fileInfo = '_totalnumcomponents-{}'.format(nComponents)

total_variance = 0
for i in range(1, nComponents):
    pca = PCA(n_components=i)
    pca.fit(data)
    PCA(copy=True, iterated_power = 'auto', n_components=i, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
    total_variance += pca.explained_variance_ratio_[i - 1]
    pylab.scatter(i, total_variance)
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph plots the total variance explained over the number of components used in PCA.')
pylab.figtext(0.02, .2, 'Maximum number of components: {}'.format(nComponents))
pylab.savefig(os.path.expanduser(graph_dir + '/pca_total_variance_vs_num_components' + fileInfo + '.pdf'), bbox_inches='tight')
