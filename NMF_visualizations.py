import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from itertools import cycle, product
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

import argparse

import matplotlib.pyplot as plt
plt.switch_backend('Agg') # this suppresses the console for plotting                                                        
import seaborn as sns

import load_kmer_cnts_jf
import warnings
from sklearn.decomposition import NMF
import _search
import _validation
import load_kmer_cnts_pasolli_jf

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/linear/'

data_sets_to_use = [
    #[['MetaHIT'], ['MetaHIT']],                                                                                                                                                                
    #[['Qin_et_al'], ['Qin_et_al']],                                                                                                                                                            
    [['Zeller_2014'], ['Zeller_2014']],                                                                                                                                                        
    #[['LiverCirrhosis'], ['LiverCirrhosis']],                                                                                                                                                  
    #[['Karlsson_2013'], ['Karlsson_2013']],                                                                                                                                                    
    #[['RA'], ['RA']],                                                                                                                                                                          
    [['Feng'], ['Feng']]
]

kmer_size = 8

for data_set in data_sets_to_use:
        data_set = data_set[0]
        kmer_dir = os.environ['HOME'] + '/deep_learning_microbiome/data/' + str(kmer_size) + 'mers_jf/'

        allowed_labels = ['0', '1']
        kmer_cnts, accessions, labelz, domain_labels = load_kmer_cnts_pasolli_jf.load_kmers(kmer_size, data_set, allowed_labels)
        print("LOADED DATASET " + str(data_set[0]) + ": " + str(len(kmer_cnts)) + " SAMPLES")
        labelz=np.asarray(labelz)
        labelz=labelz.astype(np.int)

        n_comp = [80]
        for n in n_comp:
            if n == 0:
                data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                data_normalized, labels = shuffle(data_normalized, labelz, random_state=0)
                x = data_normalized
                y = labels
            else:
                data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                data_normalized, labels = shuffle(data_normalized, labelz, random_state=0)
                V = data_normalized.T
                model = NMF(n_components = n, init='random', random_state=0, solver = 'mu', beta_loss = 'frobenius', max_iter = 1000)
                W = model.fit_transform(V)
                H = model.components_
                
                W_all = pd.DataFrame(W)
                W_all['Features'] = pd.read_csv(kmer_dir + str(kmer_size) + "mer_dictionary.gz", compression='gzip', header=None)
                meltedW = pd.melt(W_all, id_vars = "Features", var_name='Signature (i.e. Factor)', value_name='Weight')
                sns.set(style="white")
                g = sns.FacetGrid(meltedW, row = 'Signature (i.e. Factor)', sharey = True)
                g.map(sns.barplot, 'Features', 'Weight', color="blue", alpha = 0.7)
                g.set(xticklabels=[])
                plt.savefig(graph_dir + "NMF" + str(n) + "kmerProfile_" + str(data_set) + str(kmer_size) + "mers.png")
