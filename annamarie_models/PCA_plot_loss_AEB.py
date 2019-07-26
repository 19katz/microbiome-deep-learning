###My normalized PCA code on 5mers
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

#Read pickle
data = pd.read_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/loaded_5mers_174.pickle')
#print(data)

#Normalize
#also include axis=1 statement in here
data_normalized = normalize(data, axis = 1, norm = 'l1')

#Initialize figure
pylab.figure()
pylab.title("Loss versus Number of Components with 5mers")
pylab.xlabel('Number of Components')
pylab.ylabel('Mean Squared Loss')
plt.axis([0, 30, 0, 0.000000008])


#Loss versus PCA curve
no_components = 30
for i in range(1, no_components + 1):
    pca = PCA(n_components=i)
    pca.fit(data_normalized)
    data_normalized_pca = pca.transform(data_normalized)
    data_normalized_projected = pca.inverse_transform(data_normalized_pca)
    loss = ((data_normalized - data_normalized_projected)**2).mean()
    print(str(i) + ": " + str(loss))
    pylab.scatter(i, loss)

#Figure text
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph plots the mean squared loss with increasing number of PCA components.')
pylab.figtext(0.02, .2, 'Maximum number of components: {}'.format(nComponents))
