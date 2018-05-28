#!/usr/bin/env python3
#~/miniconda3/bin/python3

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

    

def load_kmers(kmer_size,data_sets, allowed_labels=['0','1']):

    kmer_cnts=[]
    accessions=[]
    labels=[]
    metadata=load_metadata()

    for data_set in data_sets:
        print(data_set)
        input_dir = os.path.expanduser('~/deep_learning_microbiome/data/%smers_jf/%s') %(kmer_size, data_set)
        file_pattern='*.gz'
        files=glob(input_dir + '/' + file_pattern)

        for inFN in files:        
            run_accession=inFN.split('/')[-1].split('_')[0]
            if run_accession in metadata.keys():
                label=metadata[run_accession]
                if label in allowed_labels:
                    labels.append(metadata[run_accession])
                    accessions.append(run_accession)

                    file = gzip.open(inFN, 'rb')
                
                    new_cnts=[]
                    for line in file:   
                        new_cnts.append(float(line.decode('utf8').strip('\n')[1:]))
                    new_cnts=np.asarray(new_cnts)
                    kmer_cnts.append(new_cnts)

    kmer_cnts=np.asarray(kmer_cnts)

    return kmer_cnts, accessions, labels
    


    

def load_metadata():
    metadata={} # run_accession -> disease_status
    directory=os.path.expanduser('~/deep_learning_microbiome/data/metadata')

    # Qin et al. T2D data:

    qin_et_al_inFN='%s/Qin_2012_ids_all.txt' %directory
    qin_et_al_file=open(qin_et_al_inFN, 'r')
    for line in qin_et_al_file:
        items=line.strip('\n').split('\t')
        run_accession=items[2]
        disease_status=items[5]
        metadata[run_accession] = disease_status

    # Zhang et al. RA data:

    RA_inFN='%s/arthritis_metaData_merged.txt' %directory
    RA_file=open(RA_inFN, 'r')
    for line in RA_file:
        items=line.strip('\n').split('\t')
        run_accession=items[2]
        disease_status=items[3]
        metadata[run_accession] = disease_status

    # MetaHIT IBD data:
    
    MetaHIT_inFN='%s/MetaHIT_ids.txt' %directory
    MetaHIT_file=open(MetaHIT_inFN, 'r')
    for line in MetaHIT_file:
        items=line.strip('\n').split('\t')
        run_accession=items[2]
        disease_status=items[5]
        metadata[run_accession] = disease_status
    

    # HMP data (everyone is healthy):
    
    HMP_inFN='%s/HMP_samples_314.txt' %directory
    HMP_file=open(HMP_inFN, 'r')
    exclude=['SRR2822459', '700034024', '700109173', '700107759', '700119226']
    for line in HMP_file:
        items=line.strip('\n').split('\t')
        run_accession=items[0]
        disease_status='0'
        if run_accession not in exclude:
            metadata[run_accession] = disease_status
    
        
    return metadata
