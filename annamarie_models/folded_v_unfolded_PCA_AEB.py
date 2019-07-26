import numpy as np
import pandas as pd
import pickle

#fold forward and reverse complement together
#source for make_complment_mer function:
##https://stackoverflow.com/questions/40952719/algorithm-to-collapse-forward-and-reverse-complement-of-a-dna-sequence-in-python
def make_complment_mer(mer_string):
    nu_mer = ""
    compliment_map = {"a" : "t", "c" : "g", "t" : "a", "g" : "c"}
    for base in mer_string:
        nu_mer += compliment_map[base]
    nu_mer = nu_mer[::-1]
    return nu_mer[:]

#get data I need
#174 rows X 1024 columns
data5 = pd.read_pickle('../loaded_5mers_174.pickle')
#print(data5)
file_names = data5.index
header = data5.columns
#get rid of file names for now so I can fold sequence columns together
data5 = data5.reset_index()
data5.drop('index', axis=1, inplace=True)

#get reverse complement of all possible 5mers
rev_comp = []
for x in header:
    rev_comp.append(make_complment_mer(x))

#Initialize new header, and grab forward and reverse
new_header = []
forward = list(header[:int(len(header)/2)])
reverse = list(rev_comp[:int(len(header)/2)])
#Fold together into new list
for i in range (0, int(len(header)/2)):
    new_header.append(forward[i] + '/' + reverse[i])
#print(new_header)

#add appropriate columns together and add to df
new_df = pd.DataFrame()
for i in range (0, int(len(header)/2)):
    #grab forward column and grab reverse column, and add all contents together
    forward_col = data5[forward[i]]
    reverse_col = data5[reverse[i]]
    collapsed = (forward_col + reverse_col)
    new_df = new_df.append(collapsed, ignore_index=True)

new_df.index = new_header
new_df.columns = file_names
new_df = new_df.T
print(new_df)

new_df.to_pickle('../folded_5mers_174.pickle')

#Folded versus not folded on 5mers
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
#folded
dataf = pd.read_pickle('../folded_5mers_174.pickle')
#5mer
data5 = pd.read_pickle('../loaded_5mers_174.pickle')


data_normalizedf = normalize(dataf, axis = 1, norm = 'l1')
data_normalized5 = normalize(data5, axis = 1, norm = 'l1')

pcaf = PCA(n_components=2)
pcaf.fit(data_normalizedf)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_newf = pcaf.transform(data_normalizedf)

pca5 = PCA(n_components=2)
pca5.fit(data_normalized5)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
data_new5 = pca5.transform(data_normalized5)

pylab.figure()

pylab.title('PC2 VS PC1')

pylab.xlabel('PC1')
pylab.ylabel('PC2')

#graph_dir = '/pollard/home/abustion/deep_learning_microbiome/analysis'

pylab.scatter(data_newf[:, 0],data_newf[:, 1], color = 'y', label = 'folded')
pylab.scatter(data_new5[:, 0],data_new5[:, 1], color = 'r', label = 'unfolded')
pylab.legend(loc='upper right')
pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph shows 5mer relative abundance data plotted with 2-component PCA')

###Loss folded versus non-folded with 5mers
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
#folded
dataf = pd.read_pickle('../folded_5mers_174.pickle')
#5mer
data5 = pd.read_pickle('../loaded_5mers_174.pickle')

#Normalize
#also include axis=1 statement in here
data_normalizedf = normalize(dataf, axis = 1, norm = 'l1')
data_normalized5 = normalize(data5, axis = 1, norm = 'l1')

#PCA w/set number of components
#pca = PCA(n_components=2)
#pca.fit(data_normalized)
#data_normalized_pca = pca.transform(data_normalized)
#data_normalized_projected = pca.inverse_transform(data_normalized_pca)
#loss = ((data_normalized - data_normalized_projected)**2).mean()
#print(loss)


#Initialize figure
pylab.figure()
pylab.title("Loss over number of components (5mers)")
pylab.xlabel('Number of Components')
pylab.ylabel('Mean Squared Loss')
plt.axis([0, 30, 0, 0.0000008])


#Loss versus PCA curve
no_components = 30
for i in range(1, no_components + 1):
    pcaf = PCA(n_components=i)
    pca5 = PCA(n_components=i)
    pcaf.fit(data_normalizedf)
    pca5.fit(data_normalized5)
    data_normalized_pcaf = pcaf.transform(data_normalizedf)
    data_normalized_pca5 = pca5.transform(data_normalized5)
    data_normalized_projectedf = pcaf.inverse_transform(data_normalized_pcaf)
    data_normalized_projected5 = pca5.inverse_transform(data_normalized_pca5)
    lossf = ((data_normalizedf - data_normalized_projectedf)**2).mean()
    loss5 = ((data_normalized5 - data_normalized_projected5)**2).mean()
#    print(str(i) + ": 3mer, " + str(loss3))
#    print(str(i) + ": 5mer, " + str(loss5))
    pylab.scatter(i, lossf, color = 'y')
    pylab.scatter(i, loss5, color = 'r')

#Figure text
axes = pylab.gca()
axes.set_ylim([0,0.00000005])



pylab.gca().set_position((.1, .4, .8, .6))
pylab.figtext(0.02, .24, 'This graph plots the mean squared loss with increasing number of PCA components.')
pylab.figtext(0.02, .2, 'Maximum number of components: {}'.format(no_components))
