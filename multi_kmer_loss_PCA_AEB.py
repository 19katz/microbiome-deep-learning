###My normalized PCA code on 3mers
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import os
import pickle

#----------Change these to 314 files-------------#
#Read pickle
#3mer
data3 = pd.read_pickle('../3mer_314_hmp.pickle')
#4m34
data4 = pd.read_pickle('../4mer_314_hmp.pickle')

#Normalize
#also include axis=1 statement in here
data_normalized3 = normalize(data3, axis = 1, norm = 'l1')
data_normalized4 = normalize(data4, axis = 1, norm = 'l1')

#Initialize figure
pylab.figure()
pylab.title("Loss over no. components (3mers vs 4mers)")
pylab.xlabel('Number of Components')
pylab.ylabel('Mean Squared Loss')
plt.axis([0, 30, 0, 0.0000018])


#Loss versus PCA curve
no_components = 30
for i in range(1, no_components + 1):
    pca3 = PCA(n_components=i)
    pca4 = PCA(n_components=i)
    pca3.fit(data_normalized3)
    pca4.fit(data_normalized4)
    data_normalized_pca3 = pca3.transform(data_normalized3)
    data_normalized_pca4 = pca4.transform(data_normalized4)
    data_normalized_projected3 = pca3.inverse_transform(data_normalized_pca3)
    data_normalized_projected4 = pca4.inverse_transform(data_normalized_pca4)
    loss3 = ((data_normalized3 - data_normalized_projected3)**2).mean()
    loss4 = ((data_normalized4 - data_normalized_projected4)**2).mean()
#    print(str(i) + ": 3mer, " + str(loss3))
#    print(str(i) + ": 5mer, " + str(loss5))
    pylab.scatter(i, loss3, color = 'b')
    pylab.scatter(i, loss4, color = 'g')


#Figure text
axes = pylab.gca()
axes.set_ylim([0,0.00000018])

pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph plots the mean squared loss with increasing number of PCA components.')
pylab.figtext(0.02, .2, 'Maximum number of components: {}'.format(no_components))
pylab.figtext(0.02, .16, '314 hmp files evaluated for each kmer size')


#####2 component
data_normalized3 = normalize(data3, axis = 1, norm = 'l1')
data_normalized4 = normalize(data4, axis = 1, norm = 'l1')

pca3 = PCA(n_components=2)
pca3.fit(data_normalized3)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_new3 = pca3.transform(data_normalized3)

pca4 = PCA(n_components=2)
pca4.fit(data_normalized4)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_new4 = pca4.transform(data_normalized4)

pylab.figure()

pylab.title('PC2 VS PC1')

pylab.xlabel('PC1')
pylab.ylabel('PC2')

#graph_dir = '/pollard/home/abustion/deep_learning_microbiome/analysis'

pylab.scatter(data_new3[:, 0],data_new3[:, 1], color = 'b', label = '3mer')
pylab.scatter(data_new4[:, 0],data_new4[:, 1], color = 'g', label = '4mer')
pylab.legend(loc='upper right')
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph shows 3mer and 4mer relative abundance data plotted with 2-component PCA')
pylab.figtext(0.02, .20, '314 hmp files evaluated for each kmer size')

#pylab.savefig(os.path.expanduser(graph_dir + '/3mer_sample_reps_pca_two_components' + '.pdf'), bbox_inches='tight')
