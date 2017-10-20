#!/usr/bin/env python

import time
import gzip
import pandas as pd
import scipy.sparse as sp
import numpy as np
import pickle
from Bio import SeqIO
from glob import glob
from itertools import product
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

##The function - this happens third
def get_kmers(fn):
    # SeqIO.parse() takes filename (fn) & format name (fastq) and returns SeqRecord iterator
    ## so we can work through file by file
    # Parse will return all the sequences in the fast.
    # .seq returns all the DNA (ignores the fasta header).
    # Will return a vector with the number of rows = number of unique seq. 
    
    kmer_gzip = gzip.open(fn, 'rt')
    seq_stuff = SeqIO.parse(kmer_gzip, 'fastq') 

    print("started appending into list loop:" + time.ctime())
    sequences = []
    x = 0
    for record in seq_stuff:
        sequences.append(str(record.seq))
        x = x + 1
        if x > 500:
            break
    print("finished appending into list loop:" + time.ctime())
    print(len(sequences))

    # Vectorization = turning collection of text documents into numerical feature vectors
    # returns a numpy array (columns = kmers, rows = frequencies)
    # returns a numpy array (with exactly 1 row taking the sum across all rows (above))
    print("started vectorizer at.." + time.ctime())

    #I know running 30 mil worked.
    cap = 30000000
    step = int(np.ceil(len(sequences)/cap))
    final = np.zeros([1,(4**kmer_size)])

    for i in range(step + 1):
        final = final + vectorizer.fit_fit_transform(sequences[(cap*i):(cap*(i+1))]).sum(axis=0)

    return(final)
    print("vectorizer finished at:" + time.ctime())
    
####This happens first
kmer_size = 3
all_kmers = [''.join(_) for _ in product(['a', 'c', 'g', 't'], repeat = kmer_size)]
print("all_kmers just finished setting at: " + time.ctime())

####This happens second
# create vectorizer class (this is an object):
# seems equivalent to CountVectorizer bc use_idf = False and norm = None
# idea for later-- try out Marisa-trie again but with CountVectorizer instead of TfidfVectorizer
vectorizer = TfidfVectorizer(
    # count at level of character rather than word
    analyzer = 'char',
    # can set range (min, max)
    ngram_range = (kmer_size, kmer_size),
    vocabulary = all_kmers,
    # False makes idf similar to CountVectorizer
    use_idf = False,
    # could also be 'l1' or 'l2'
    norm = None
)
print("vectorizer just finished setting:" + time.ctime())

# -1 means that you use all the CPUs (be very careful)
# instead start with 1 core, then scale up to 2-4 if taking up too much RAM
# glob -- grab all *fa.gz files in the current dir (change to fastq), can also specify a dir (glob('home/dir/path/*fastq.gz')).
# Can also do this recursively (glob('home/dir/path/**fastq.gz', recursive=True))
# delayed(get_kmers)(fn) sets up 'asynchronous' function so that once job is done, that core is filled iwth another job.  
# kmers is a list of numpy objects generated by get_kmers

##This then calls get_kmers function
kmers = Parallel(n_jobs = 1)(delayed(get_kmers)(fn) for fn in glob('*fastq.gz'))

##This happens fourth
# create pandas dframe
sparse_kmers_df = pd.DataFrame(
    # stack the arrays vertically into a sparse (sp) array
    np.vstack(kmers),
    # naming all the columns with the kmers from the vectorizer
    columns = all_kmers,
)

#will tell you how much memory etc. 
#sparse_kmers_df.info()
#print()

#pickled dataframe
sparse_kmers_df.to_pickle('/pollard/home/abustion/deep_learning_microbiome/data/sample_reps_' + str(kmer_size) + '.pickle')

#how to read later
#pd.read_pickle('/pollard/home/abustion/play/pickles/fullfinalstack.pickle')

print("finished pickling:" + time.ctime())
