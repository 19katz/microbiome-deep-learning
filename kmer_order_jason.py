#!/usr/bin/env python3

import gzip
import pandas as pd
import numpy as np
import ntpath

from Bio import SeqIO
from glob import glob
from itertools import product

from functools import partial
from multiprocessing import Pool
import os.path

file_pattern='*10mers.txt'
input_dir = os.path.expanduser('~/deep_learning_microbiome/data/10mers/HMP')
files=glob(input_dir + '/' + file_pattern)

Ten_mers=[]

for inFN in files:
    file = open(inFN,"r")
    if len(Ten_mers) ==0:
        for line in file:
            Ten_mers.append(line.strip('\n').split()[0])
    else:
        index=0
        for line in file:   
            string=line.strip('\n').split()[0]
            #print(string +'\t' +Ten_mers[index])
            if Ten_mers[index]!=string:
                print(index)
            index +=1



