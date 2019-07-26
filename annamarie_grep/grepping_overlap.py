import os
import pandas as pd
import numpy as np
import csv
import itertools
from itertools import chain
import gzip
import re

os.chdir('/pollard/home/ngarud/shattuck/metagenomic_fastq_files/LiverCirrhosis/combined_fastq_files/')

#change for whatever kmers you're interested in; these are the top ten weighted kmers from Signature 0 of LiverCirrhosis 8mers (NMFk=10)
kmer_list = ['AAAAAATA', 'ATAAAAAA', 'AAATAAAA', 'AAAAATAT', 'AAAATAAA', 'AATAAAAA', 'AAAAATAA', 'AAAGAAAA', 'AAAAGAAA', 'AAAAAAAT']

def get_kmer_cnts(kmer):
    kmer_cnts = 0
    with gzip.open('LD-91_2.fastq.gz', 'rt') as f:
        for line in f:
            r = re.findall(kmer,line)
            if len(r) > 0:
                kmer_cnts = kmer_cnts + 1
    return kmer_cnts

def overlaps1(kmer1, kmer2):
    overlap_list = []
    for i in range(1, min(len(kmer1), len(kmer2))):
        if kmer1[-i:] == kmer2[:i]:
            overlap_list.append(kmer1 + kmer2[i:])
    return overlap_list

def overlaps2(kmer1, kmer2):
    overlap_list = overlaps1(kmer1,kmer2) + overlaps1(kmer2,kmer1)
    return overlap_list

df = pd.DataFrame(columns = ['pair','kmer1', 'kmer2', 'overlaps'])
i = 0
for pair in itertools.combinations(kmer_list, 2):
    i = i + 1
    print("Hi, we're on loop " + str(i))
    kmer1 = pair[0]
    print(kmer1)
    kmer2 = pair[1]
    print(kmer2)
    overlaps = overlaps2(pair[0], pair[1])
    print(overlaps)
    
    kmer1_cnts = get_kmer_cnts(kmer1)
    print(kmer1_cnts)
    kmer2_cnts = get_kmer_cnts(kmer2)
    print(kmer2_cnts)
    overlaps_cnts = 0
    print("okay now looping through overlaps")
    for kmer in overlaps:
        overlaps_cnts = get_kmer_cnts(kmer) + overlaps_cnts
        print("for now: " + str(overlaps_cnts))
        
    print(kmer1 + ': ' + str(kmer1_cnts) + '\t' + 
          kmer2 + ': ' + str(kmer2_cnts) + '\t' + 
          'overlaps: ' + str(overlaps_cnts))
    
    df = df.append(pd.DataFrame([[pair[0] + ":" + pair[1], kmer1_cnts, kmer2_cnts, overlaps_cnts]], columns=['pair','kmer1', 'kmer2', 'overlaps']))

df.to_csv("/pollard/home/abustion/deep_learning_microbiome/analysis/NMF/grepping_analysis/LC8mers10k.csv")
