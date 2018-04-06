#!/usr/bin/env python3
#~/miniconda3/bin/python3

import gzip
import pandas as pd
import numpy as np
import sys
from glob import glob
import os.path

sample=sys.argv[1]
kmer_size=sys.argv[2]
data_set=sys.argv[3]

dictionary_FN=gzip.open(os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%smer_dictionary.gz' %(kmer_size, kmer_size)),'rb')

dictionary=[]
dictionary_count={}
for line in dictionary_FN:
    line=line.decode('utf8').strip('\n')
    dictionary.append(line)
    dictionary_count[line]='>0\n'

# read in the file: 
inFN=os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s/tmpOut.txt' % (kmer_size, data_set)) 
inFile=open(inFN, 'r')
outFN=os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s/%s_%smer.gz' % (kmer_size, data_set, sample,kmer_size))
outFile=gzip.open(outFN, 'wb')    

kmer=''
for line in inFile: 
    if line[0]!='>':
        kmer=line.strip('\n')
    else:
        value=line
        dictionary_count[kmer]=value

# iterate through the dictionary and output

print(len(dictionary))
for i in range(0, len(dictionary)):
    kmer=dictionary[i]
    num=dictionary_count[kmer]
    outFile.write(( num ).encode())
