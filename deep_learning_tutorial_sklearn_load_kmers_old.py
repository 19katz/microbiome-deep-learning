# always run miniconda for keras:
# ./miniconda3/bin/python

import numpy as np
from numpy import random
import pandas as pd
import os
import pylab
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO
import gzip


# using biopython, load the fastq file
#fastq_dict = SeqIO.index("/home/ngarud/deep_learning/deep_learning_data/SRR038746_1_test.fastq", "fastq")

#for sequence in fastq_dict:
#    print(fastq_dict[sequence].seq)


# try iterating through a fastq.gz file
#fasta_dict={}
#file=gzip.open('/home/ngarud/deep_learning/deep_learning_data/SRR038746_1.fastq.gz','rb')
#read_sequence=False
#for line in file:
#    if read_sequence==True:
#        fasta_dict[header]=line.strip()
#        read_sequence=False
#    elif line[0]=='@':
#        header=line.strip()
#        read_sequence=True

# try iterating through a fastq file
fasta_dict={}
file=open('/home/ngarud/deep_learning/deep_learning_data/SRR038746_1_test.fastq','r')
read_sequence=False
for line in file:
    if read_sequence==True:
        fasta_dict[header]=line.strip()
        read_sequence=False
    elif line[0]=='@':
        header=line.strip()
        read_sequence=True


sequences_df=pd.DataFrame(pd.Series(fasta_dict))
v = CountVectorizer(analyzer = 'char', ngram_range = (7, 7), lowercase = True)

kmers_df = pd.DataFrame(
    v.fit_transform(sequences_df[0]).todense(),
    index = sequences_df[0].index,
    columns = v.get_feature_names()
)
