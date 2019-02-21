import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import nimfa
import argparse

# import private scripts                                                                                                                                                                                 
import load_kmer_cnts_jf
import stats_utils_AEB

#plotting                                                                                                                                                                                                
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # this suppresses the console for plotting                                                                                                                                     
import seaborn as sns

#Figure function
def pick_rank_vis(data, max_factor, title):
    V = data.T
    nmf = nimfa.Nmf(V, max_iter=1000000, update='euclidean', rank=2, track_error=True)
    r = nmf.estimate_rank(rank_range=range(2,max_factor))
    
    result_array = []
    for rank, vals in r.items():
        result_array.append([rank, vals['rss'], vals['cophenetic']])
    df = pd.DataFrame(result_array, columns=['rank', 'rss', 'coph'])
    
    fig, ax1 = plt.subplots()
    plt.xlabel('Number of Kmer signatures')
    ax2 = ax1.twinx() 
    ax1.set_ylabel('Cophenetic correlation coefficient', color = 'lightsalmon')
    ax2.set_ylabel('RSS', color = 'cadetblue')
    
    for i in df.iterrows():
        coph = df['coph']
        recon_err = df['rss']
        rank = df['rank']
        ax1.plot(df['rank'], coph, color = 'lightsalmon')
        ax2.plot(df['rank'], recon_err, color = 'cadetblue')

    plt.savefig("/pollard/home/abustion/deep_learning_microbiome/analysis/NMF/alexandrov" +
                str(title) +
                "_011418.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Program to create NMF recon/stab figures")
    parser.add_argument('-ds', type = str, default = 'LiverCirrhosis', help = "Data set")
    parser.add_argument('-k', type = int, default = 5, help = "Kmer Size")

    arg_vals = parser.parse_args()
    data_set = arg_vals.ds
    kmer_size = arg_vals.k

data_set = [data_set]
allowed_labels = ['0', '1']
kmer_cnts, accessions, labelz, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size,
                                                                            data_set,
                                                                            allowed_labels)

print("LOADED DATASET " + str(data_set[0]) + ": " + str(len(kmer_cnts)) + " SAMPLES")
labelz=np.asarray(labelz)
labelz=labelz.astype(np.int)
data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')

pick_rank_vis(data=data_normalized, max_factor=100, title = (str(data_set) + str(kmer_size)))
