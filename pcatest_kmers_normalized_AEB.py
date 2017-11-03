import csv
import bz2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
import pylab
import os
import pickle

#Read pickle
data = pd.read_pickle('/pollard/home/abustion/play/pickles/jf.pickle')

#Normalize
#also include axis=1 statement in here
#maybe the below, but double check
data_normalized = normalize(data, axis = 1, norm = 'l1')

#What I tried before
#data_normalized = normalize(data, norm='l1')

pca = PCA(n_components=2)
pca.fit(data_normalized)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_new = pca.transform(data_normalized)
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
pylab.savefig(os.path.expanduser(graph_dir + '/3mer_pca_two_components_normalized' + '.pdf'), bbox_inches='tight')


pylab.figure()
pylab.title('Explained Variance vs Number of Components')
pylab.xlabel('Components')
pylab.ylabel('Total Variance')

nComponents = 30
fileInfo = '_totalnumcomponents-{}'.format(nComponents)

total_variance = 0
for i in range(1, nComponents):
    pca = PCA(n_components=i)
    pca.fit(data_normalized)
    PCA(copy=True, iterated_power = 'auto', n_components=i, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
    total_variance += pca.explained_variance_ratio_[i - 1]
    pylab.scatter(i, total_variance)
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph plots the total variance explained over the number of components used i\
n PCA.')
pylab.figtext(0.02, .2, 'Maximum number of components: {}'.format(nComponents))
pylab.savefig(os.path.expanduser(graph_dir + '/3mer_pca_total_variance_vs_num_components_norm' + fileInfo + '.pdf'), box_inches='tight')
