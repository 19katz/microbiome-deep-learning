#!/usr/bin/env python3
#~/miniconda3/bin/python3

import gzip
import pandas as pd
import numpy as np
from numpy import array
import ntpath

from Bio import SeqIO
from glob import glob
from itertools import product

from functools import partial
from multiprocessing import Pool
import os.path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import load_kmer_cnts_Katherine_jf


def load_kmers(kmer_size,data_sets, allowed_labels):

    kmers_cnts = []
    accessions = []
    labels = []
    domain_labels = []
    for data_set in data_sets:
        kmer_cnts, labels = load_kmer_cnts_Katherine_jf.load_kmers(kmer_size, [data_set])
    for i in range(len(labels)):
        labels[i] = labels[i][0]
    return kmer_cnts, accessions, labels, domain_labels
    

def onehot_encode(labels):
    
    values = array(labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded



if __name__ == "__main__":
    for kmer_size in [5]:
        #kmers, labels = load_kmers(kmer_size, ['HMP', 'Feng', 'Zeller_2014', 'RA', 'MetaHIT','LiverCirrhosis', 'Karlsson_2013', 'Qin_et_al'])
        for dataset in ['HMP', 'Feng', 'Zeller_2014', 'RA', 'MetaHIT','LiverCirrhosis', 'Karlsson_2013', 'Qin_et_al']:
            kmers, accessions, labels, domain_labels = load_kmers(kmer_size, [ dataset, ], ['0', '1'])
            total = kmers.shape[0]
            diseased = [labels[i][0] for i in range(len(labels))].count('1')
            print(dataset + ": " + str(total) + " samples, " + str(total - diseased) + " control, " + str(diseased) + " diseased")
