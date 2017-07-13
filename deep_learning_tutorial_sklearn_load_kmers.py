#!/usr/bin/env python

import gzip
import pandas as pd
import scipy.sparse as sp

from Bio import SeqIO
from glob import glob
from itertools import product
# joblib allows you to run multiple parllel jobs from python.
from sklearn.externals.joblib import Parallel, delayed 
from sklearn.feature_extraction.text import TfidfVectorizer

def get_kmers(fn):
    # fn = abbrev for file name
    # parse will return all the sequences in the fast. .seq returns all the DNA (ignores the fasta header). Will return a vector with the number of rows = number of unique seq. 
    sequences = [str(_.seq) for _ in SeqIO.parse(gzip.open(fn, 'rt'), 'fastq')]

    # return vectorizer.fit_transform(sequences) # returns a numpy array (columns = kmers, rows = sequencies)
    return vectorizer.fit_transform(sequences).sum(axis=0) # returns a numpy array (with exactly 1 row taking the sum across all rows (above))


kmer_size = 5
all_kmers = [''.join(_) for _ in product(['a', 'c', 'g', 't'], repeat = kmer_size)]

# create vectorizer class (this is an object):
vectorizer = TfidfVectorizer(
    analyzer = 'char', # count at level of character instead of word. 
    ngram_range = (kmer_size, kmer_size), # can set a range (min, max). The longer kmer is, the less accurate the predicition is. 
    vocabulary = all_kmers,
    use_idf = False, # false makes idf similar to countVectorizer. # TFidf -- frequency of word vs document frequency (i.e. 'the' and 'and'). May not want to do this. 
    #norm = None # scale vector (L1 or L2 norm)
    norm='l1' # change this to None or l2 later?
)

# -1 means that you use all the CPUs (be very careful, because if one CPU takes a lot of ram, then you can really swamp system. instead start with 1 core, guesstimate how much RAM it will take, then scale up. May make sense to use 2-4 cores)
# glob -- grab all *fa.gz files in the current dir (change to fastq), can also specify a dir (glob('home/dir/path/*fastq.gz')). Can also do this recursively (glob('home/dir/path/**fastq.gz', recursive=True))
# delayed(get_kmers)(fn). Sets up 'asynchronous' function whereby once a job is done, that core is filled iwth another job.  
# kmers is a list of numpy objects generated by get_kmers

kmers = Parallel(n_jobs = -1)(delayed(get_kmers)(fn) for fn in glob('*fa.gz'))

# create pandas dframe
sparse_kmers_df = pd.SparseDataFrame(
    # stack the arrays vertically into a sparse (sp) array
    sp.vstack(kmers),
    # naming all the columns with the kmers from the vectorizer
    columns = all_kmers,
    # if values are not present, then fill with 0
    default_fill_value = 0
)
print(sparse_kmers_df.head())

# will tell you how much memory etc. 
sparse_kmers_df.info()
print()
# to_dense will un-sparse it. Sometime sparse doesn't work with some functions. 
sparse_kmers_df.to_dense().info()
