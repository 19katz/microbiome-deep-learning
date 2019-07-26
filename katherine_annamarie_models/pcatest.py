import csv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas


data = np.zeros((1573, 5952), dtype=float)

################
# reading in data with for loop
################

'''
col = 0
firstline = True
with open("relative_abundance.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        if not firstline:
            data[:, col] = line[1:]
            col += 1
        firstline = False
'''

################
# read in data with pandas
################

data = pandas.read_csv("relative_abundance.txt", sep = '\t', index_col = 0).T 


#print(data)
pca = PCA(n_components=2)
pca.fit(data)
PCA(copy=True, iterated_power = 'auto', n_components=2, random_state=None,
    svd_solver = 'auto', tol=0.0, whiten = False)
print(pca.explained_variance_ratio_)
#print(pca.components_)
data_new = pca.transform(data)
print(data_new)

fig = plt.figure()
fig.suptitle('PCA Relative Abundance')

ax = fig.add_subplot(111)
ax.set_title('PC2 VS PC1')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

ax.scatter(data_new[:,0],data_new[:,1])
plt.show()

