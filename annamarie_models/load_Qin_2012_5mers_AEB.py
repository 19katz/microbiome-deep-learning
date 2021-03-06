import random as rn
import pandas as pd
import numpy as np
from itertools import product
from glob import glob
import re
import os.path
from os.path import basename
import pickle
import csv

# Load the given list of 5mer count files, can take a list of ids to skip.
# If a file name has _1 in it, it will automatically load the corresponding _2 file
# and merge their kmer counts.
def load_kmer_cnts_from_files(files, filters=[], load_1_only=False):
    kmers_df = pd.DataFrame()
    for f in files:
        #print("Processing: " + f)
        drop = False
        for filter in filters:
            if filter in f:
                # sample id to be skipped is in the file name
                drop = True
                break
        if drop:
            continue
        kmers_cnt = pd.read_csv(f, sep='\t', header=None)
        if not load_1_only:
            s = re.sub(r'/([^/_]+)_1\.fastq\.gz\.5mer\.rawcnt$', r'/\1_2.fastq.gz.5mer.rawcnt', f)
            if s != f and os.path.exists(s):
                # The file name has _1 in it and the _2 file exists, so load the _2 file and add its counts
                # to the _1 file's. This is element-wise matrix addition.
                kmers_cnt = kmers_cnt.add(pd.read_csv(s, sep='\t', header=None))
        # This is matrix append - it creates a new matrix
        kmers_df = kmers_df.append(kmers_cnt)

    return kmers_df

# These two T2D samples have no metadata
t2d_filter = ['SRR413693', 'SRR413722']
def load_kmer_cnts_for_t2d(shuffle=False, kmer_cnts_dir = '../data_generated', filter=True, load_1_only=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    kmers_df = load_kmer_cnts_from_files(
        glob(kmer_cnts_dir + '/'  + 'SRR341*_1.fastq.gz.5mer.rawcnt') +
        glob(kmer_cnts_dir + '/'  + 'SRR413*_1.fastq.gz.5mer.rawcnt'), filters=(t2d_filter if filter else []), load_1_only=load_1_only)
    if shuffle:
        for r in kmers_df.values:
            # shuffle the counts within each sample (row of the matrix)
            np.random.shuffle(r)
    return kmers_df

def load_kmer_cnts_for_t2d_with_labels(kmer_cnts_dir = '/pollard/home/ngarud/deep_learning_microbiome/data/5mer_data_katherine', data_dir='/pollard/home/abustion/deep_learning_microbiome/data/metafiles', filter=True, load_1_only=False, shuffle_labels=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    t2d_files = glob(kmer_cnts_dir + '/'  + 'SRR341*_1.fastq.gz.5mer.rawcnt') +\
               glob(kmer_cnts_dir + '/'  + 'SRR413*_1.fastq.gz.5mer.rawcnt')
    kmers_df = load_kmer_cnts_from_files(t2d_files, filters=(t2d_filter if filter else []), load_1_only=load_1_only)

    # map of sample id to labels
    dict = {}
    
    # get sample labels from metadata file
    metadata = pd.read_csv(data_dir +'/' + 'Qin_2012_ids_all.txt', sep='\t', header=None)
    for v in metadata.values:
        dict[v[2]] = [v[5], 'Asia', v, 'China', 'T2D', 'T2D', 'T2D' if v[5] == '1' else 'Healthy']
    labels = []
    for f in t2d_files:
        filtered = False
        for fltr in (t2d_filter if filter else []):
            if fltr in f:
                # sample id to be skipped is in the file name
                filtered = True
                break
        if not filtered:
            # append the asosciated label for the sample to the label list at the same position as its 
            # kmer count row is in the DataFrame
            labels.append(dict[re.search(r'.*/([^/_]+)_1\.fastq\.gz\.5mer\.rawcnt$', f).groups()[0]])
    if shuffle_labels:
        # shuffle the labels - note that we don't change the distribution of the class labels
        rn.shuffle(labels)

    #Add in kmer ids in a header row
    kmer_size = 5
    all_kmers = [''.join(_) for _ in product(['a', 'c', 'g', 't'], repeat = kmer_size)]
    kmers_df.columns = [all_kmers]

    #Add in file names in a header column
    no_path_files = [os.path.basename(x).split('_')[0] for x in t2d_files]
    kmers_df.index = no_path_files
    
    ##my addition
    kmers_df.to_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/Qin_5mers.pickle')
    print(kmers_df)
    
    disease_labels = pd.DataFrame(labels)
    disease_labels.to_csv('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/Qin_label.csv', index=False, header=False)
#    with open("/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/Qin_label.csv", "w", newline='') as f:
#        writer = csv.writer(f)
#        writer.writerows(labels)
    ##end my addition
    
    return kmers_df, labels

load_kmer_cnts_for_t2d_with_labels()
