#general
import os
import pandas as pd
import numpy as np
import pickle

#outside functions
import load_kmer_cnts_pasolli_jf

#data
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF

#plotting
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # this suppresses the console for plotting                                                        
import seaborn as sns

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/linear/'

def vis_NMF(data, n_components=5, init = 'random', solver='mu', beta_loss='frobenius', max_iter=1000, random_state=0, sort = False, title = "dataset"):
    model = NMF(
        n_components = n_components,
        init = init,
        solver = solver, 
        beta_loss = beta_loss,
        max_iter = max_iter, 
        random_state = random_state
    )
    
    #NMF matrixes
    V = data.T
    W = model.fit_transform(V)
    H = model.components_
    print(W.shape)
    print(H.shape)
    
    #Getting it ready for plotting
    W_all = pd.DataFrame(W)
    W_all['Features'] = pd.read_csv(kmer_dir + str(kmer_size) + "mer_dictionary.gz", compression='gzip', header=None)
    
    #Plotting
    if sort == False:
        meltedW = pd.melt(W_all, id_vars = "Features", var_name='Signature (i.e. Factor)', value_name='Weight')
        sns.set(style="white")
        g = sns.FacetGrid(meltedW, row = 'Signature (i.e. Factor)', sharey = True, size = 7)
        g.map(sns.barplot, 'Features', 'Weight', color="blue", alpha = 0.7)
        g.set(xticklabels=[])
        plt.subplots_adjust(top=.93)
        g.fig.suptitle(title)
        plt.savefig(graph_dir + '101018_' + str(data_set) + str(kmer_size) + 'mers'  + str(n_components) + 'factors.png')
        
    elif sort == True:
        meltedW = pd.melt(W_all, id_vars = "Features", var_name='Signature (i.e. Factor)', value_name='Weight').sort_values(by = 'Weight')
        meltedW.to_pickle("/pollard/home/abustion/deep_learning_microbiome/data_AEB/pickled_dfs/" + '101018_' + str(data_set) + str(kmer_size) + 'mers'  + str(n_components) + 'factors.pickle')
        sns.set(style="white")
        g = sns.FacetGrid(meltedW, row = 'Signature (i.e. Factor)', sharey = True, size = 7)
        g.map(sns.pointplot, 'Features', 'Weight', color='purple', alpha = 0.7)
        g.set(xticklabels=[])
        plt.subplots_adjust(top=.93)
        g.fig.suptitle(title)
        plt.savefig(graph_dir + '101018_' + str(data_set) + str(kmer_size) + 'mers'  + str(n_components) + 'factors.png')

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

        data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
        data_normalized, labels = shuffle(data_normalized, labelz, random_state=0)
        
        vis_NMF(data_normalized, n_components=80, sort = True, title=(str(data_set) + str(kmer_size) + 'mers'  + str(80) + 'factors'))

