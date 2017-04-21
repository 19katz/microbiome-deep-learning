# always run miniconda for keras:
# ./miniconda3/bin/python

import numpy as np
from numpy import random
import pandas as pd
import os
#import pylab
from sklearn.feature_extraction.text import CountVectorizer
#from Bio import SeqIO


# using biopython, load the fastq file
#fasta_dict = SeqIO.index("/home/ngarud/deep_learning/deep_learning_data/SRR038746_1_test.fasta", "fasta")

#for sequence in fasta_dict:
#    print(sequence)
#fasta_dict["SRR038746.22"].seq

#data=['NCATAGACATCAAGATTTTCTATAGTCTTTCCGTTACGCACCAACTTTATGCCACAT']
sequences_df=pandas.read_table("~/Documents/project_data/deepLearning/sample.fasta",index_col=0,header=None)
sequences_df.columns=["sequence"]
v = CountVectorizer(analyzer = 'char', ngram_range = (2, 5), lowercase = True)
kmers_df = pd.DataFrame(
    v.fit_transform(sequences_df['sequence']).todense(),
    index = sequences_df.index,
    columns = v.get_feature_names()
)

# Sean's code
#v = CountVectorizer(analyzer = 'char', ngram_range = (1, 4), lowercase = True)
#kmers_df = pd.DataFrame(
#    v.fit_transform(sequences_df['sequence']).todense(),
#    index = sequences_df.index,
#    columns = v.get_feature_names()
#)


