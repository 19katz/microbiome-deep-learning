#!/usr/bin/env python3

import random as rn
import pandas as pd
import numpy as np
from itertools import product
from glob import glob
import re
import os.path

"""
The load functions without labels are for autoencoder experiments only, and the ones with labels
are for supervised learning. The kmer counts are returned as a DataFrame with samples as rows and
kmers as columns.

The labels are returned as a Python list (not numpy array) of [<diseased - '1' or '0'>, <continent>,
<list of original fields in the metadata file for this sample>, <country>, <disease name in the data source - HMP for HMP>,
<data source name>, <name of disease for this sample - 'Healthy' if none>, <BMI (only for MetaHIT)> ]

The label list per sample evolved gradually, and to not break old code, it's only appended with new
fields - existing labels were never changed, even if some of them are no longer used.
"""

kmer_size = 5
all_kmers_caps = [''.join(_) for _ in product(['A', 'C', 'G', 'T'], repeat = kmer_size)]

# Loads the HMP files - for the very first autoencoder experiments, no longer used 
def load_kmer_cnts(filename_pattern='700*_1.fastq.gz.5mer.rawcnt',
                   kmer_cnts_dir = '../data_generated', load_1_only=True):
    return load_kmer_cnts_from_files(glob(kmer_cnts_dir + '/' + filename_pattern), load_1_only=load_1_only)

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

# List of unique HMP gut sample ids from Nandita
hmp_no_timedup_ids = [
    '700100022',
    "700113954",
    "700023788c",
    "700024509",
    "700015181",
    "700111439",
    "700163030",
    "700171066",
    "700024930",
    "700024233c",
    "700113975c",
    "700113762",
    "700117172c",
    "700102659",
    "700164050",
    "700023578",
    "700033363",
    "700015981",
    "700023919c",
    "700106170c",
    "700102299c",
    "700023845",
    "700109987",
    "700032068",
    "700116505",
    "700024673",
    "700116468",
    "700102043",
    "700171114",
    "700023634",
    "700021306",
    "700035861c",
    "700106876",
    "700171648",
    "700014562",
    "700110222",
    "700117000c",
    "700015245c",
    "700095524",
    "700116668",
    "700096865",
    "700035785c",
    "700111505",
    "700117031c",
    "700023872",
    "700111222c",
    "700037852c",
    "700021824",
    "700164450",
    "700016456c",
    "700038072",
    "700034622",
    "700111156",
    "700038386",
    "700014954",
    "700109921",
    "700099886",
    "700172726",
    "700034926c",
    "700037042c",
    "700171115",
    "700173483",
    # outlier
    "700116730c",
    "700035533",
    "700107189c",
    "700024086",
    "700171441",
    "700015415c",
    "700021876",
    "700037738c",
    "700109449",
    # outlier
    "700171390",
    "700034081",
    "700096700",
    "700096380c",
    "700165634",
    "700161856",
    "700111296c",
    "700023267",
    "700014837",
    "700035157",
    "700116028",
    "700101638c",
    "700038761c",
    "700024866c",
    "700016960",
    "700171324",
    "700015113c",
    "700038053",
    "700033435",
    # outlier
    "700035237",
    "700033989c",
    "700024449",
    "700016610c",
    "700110354",
    "700095717",
    "700016210",
    "700015922",
    "700117069c",
    "700032222c",
    "700023337",
    "700037123c",
    # outlier
    "700099512c",
    "700024711c",
    "700117755",
    "700038806c",
    "700095213c",
    "700116568",
    "700107547c",
    "700032338",
    "700033502",
    "700014724",
    "700116917",
    # outlier
    "700098429c",
    "700095486c",
    "700172498",
    "700033665",
    "700037868",
    "700015857",
    "700033201",
    "700023113",
    "700035373c",
    "700106291c",
    "700032133c",
    "700033153c",
    "700106056",
    "700021902c",
    "700034838",
    "700106198c",
    "700106065c",
    "700116865",
    "700116148",
    "700095831c",
    "700024437",
    "700098669",
    "700164339c",
    "700032413",
    "700099307",
    "700037284c",
    "700034254c",
    "700111745",
    "700117682",
    "700096047",
    "700097688",
    "700117766",
    "700016542c",
    "700097906c",
    "700165778",
    "700173023",
    "700038870",
    "700016142c",
    "700121639",
    "700163868",
    "700015702",
    "700117938",
    "700113867",
    "700034166c",
    "700106465c",
    "700038414c",
    "700108341",
    "700024545",
    "700024752c",
    "700035747c",
    "700016716c",
    "700034794c",
    "700033922c",
    "700106229c",
    "700038594",
    "700116401",
    "700024318",
    "700095647c",
    "700032944c",
    "700024024",
    "700024998c",
    "700101534",
    "700016765",
    "700033797",
    "700013715",
    "700024615",
]

# HMP sample outliers
hmp_ids_to_be_filtered = [ "700116730c", "700171390", "700035237", "700099512c", "700098429c"]
def load_kmer_cnts_for_hmp(shuffle=False, kmer_cnts_dir = '../data_generated', filter=True, load_1_only=False):
    files = []
    for id in hmp_no_timedup_ids:
        if (not filter) or not (id in hmp_ids_to_be_filtered):
            # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
            files.append(kmer_cnts_dir + '/' + id + "_1.fastq.gz.5mer.rawcnt")
    kmers_df = load_kmer_cnts_from_files(files, load_1_only=load_1_only)
    if shuffle:
        for r in kmers_df.values:
            # shuffle the counts within each sample (row of the matrix)
            np.random.shuffle(r)
    return kmers_df

def load_kmer_cnts_for_hmp_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    files = []
    for id in hmp_no_timedup_ids:
        if (not filter) or not (id in hmp_ids_to_be_filtered):
            # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
            files.append(kmer_cnts_dir + '/' + id + "_1.fastq.gz.5mer.rawcnt")
    kmers_df = load_kmer_cnts_from_files(files, load_1_only=load_1_only)
    # For HMP, shuffling labels or not doesn't matter because everyone's labels are the same -- healthy, US, and NA
    return kmers_df, [ ['0', 'North America', None, 'United States', 'HMP', 'HMP', 'Healthy'] for i in range(len(kmers_df.values)) ]

# MetaHIT outlier                    
metahit_filter = ['ERR011293']
def load_kmer_cnts_for_metahit(shuffle=False, kmer_cnts_dir = '../data_generated', filter=True, load_1_only=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    kmers_df = load_kmer_cnts_from_files(
        glob(kmer_cnts_dir + '/'  + 'ERR011*_1.fastq.gz.5mer.rawcnt'), filters=(metahit_filter if filter else []), load_1_only=load_1_only)
    if shuffle:
        for r in kmers_df.values:
            # shuffle the counts within each sample (row of the matrix)
            np.random.shuffle(r)
    return kmers_df

def load_kmer_cnts_for_metahit_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    metahit_files = glob(kmer_cnts_dir + '/'  + 'ERR011*_1.fastq.gz.5mer.rawcnt')
    kmers_df = load_kmer_cnts_from_files(metahit_files, filters=(metahit_filter if filter else []), load_1_only=load_1_only)

    # map of sample id to labels
    dict = {}
    
    # get sample labels from metadata file
    metadata = pd.read_csv(data_dir +'/' + 'MetaHIT_ids.txt', sep='\t', header=None)
    for v in metadata.values:
        dict[v[2]] = [v[5], 'Europe', v, v[3], 'IBD', 'MetaHIT', 'IBD' if v[5] == '1' else 'Healthy']
    labels = []
    for f in metahit_files:
        filtered = False
        for fltr in (metahit_filter if filter else []):
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
    return kmers_df, labels


def load_metahit_with_obesity_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    metahit_files = glob(kmer_cnts_dir + '/'  + 'ERR011*_1.fastq.gz.5mer.rawcnt')
    kmers_df = load_kmer_cnts_from_files(metahit_files, filters=(metahit_filter if filter else []), load_1_only=load_1_only)

    # map of sample id to labels
    dict = {}
    
    # get sample labels from metadata file
    metadata = pd.read_csv(data_dir +'/' + 'MetaHIT_ids.txt', sep='\t', header=None)
    for v in metadata.values:
        dict[v[2]] = ['1' if v[6] == 'obese' else '0', 'Europe', v, v[3], 'Obese', 'MetaHIT', 'Obese' if v[6] == 'obese' else 'Healthy']
    labels = []
    for f in metahit_files:
        filtered = False
        for fltr in (metahit_filter if filter else []):
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
    return kmers_df, labels

def load_metahit_with_bmi_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    metahit_files = glob(kmer_cnts_dir + '/'  + 'ERR011*_1.fastq.gz.5mer.rawcnt')
    kmers_df = load_kmer_cnts_from_files(metahit_files, filters=(metahit_filter if filter else []), load_1_only=load_1_only)

    # map of sample id to labels
    dict = {}
    
    # get sample labels from metadata file
    metadata = pd.read_csv(data_dir +'/' + 'MetaHIT_ids.txt', sep='\t', header=None)
    for v in metadata.values:
        dict[v[2]] = [v[5], 'Europe', v, v[3], 'Obese', 'MetaHIT', 'IBD' if v[5] == '1' else 'Healthy', v[6] ]
    labels = []
    for f in metahit_files:
        filtered = False
        for fltr in (metahit_filter if filter else []):
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
    return kmers_df, labels

def load_kmer_cnts_for_ra(shuffle=False, kmer_cnts_dir = '../data_generated', load_1_only=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    kmers_df = load_kmer_cnts_from_files(
        glob(kmer_cnts_dir + '/'  + 'ERR589*_1.fastq.gz.5mer.rawcnt'), load_1_only=load_1_only)
    if shuffle:
        for r in kmers_df.values:
            # shuffle the counts within each sample (row of the matrix)
            np.random.shuffle(r)
    return kmers_df

# Load the RA data with labels, with the option of considering only those with high severity T2D and no treatment as diseased
def load_kmer_cnts_for_ra_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, high_no_treat=False, shuffle_labels=False):
    # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
    ra_files = glob(kmer_cnts_dir + '/'  + 'ERR589*_1.fastq.gz.5mer.rawcnt')
    kmers_df = load_kmer_cnts_from_files(ra_files, load_1_only=load_1_only)

    # map of sample id to labels
    dict = {}
    
    # get sample labels from metadata file
    metadata = pd.read_csv(data_dir +'/' + 'arthritis_metaData_merged.txt', sep='\t', header=None)
    for v in metadata.values:
        sick = v[3]
        if high_no_treat:
            sick = ('1' if (v[4] == 'high' and v[5] == '0') else '0')
        dict[v[2]] = [sick, 'Asia', v, 'China', 'RA', 'RA', 'RA' if v[3] == '1' else 'Healthy']
    labels = []
    for f in ra_files:
        # append the asosciated label for the sample to the label list at the same position as its 
        # kmer count row is in the DataFrame
        labels.append(dict[re.search(r'.*/([^/_]+)_1\.fastq\.gz\.5mer\.rawcnt$', f).groups()[0]])
    if shuffle_labels:
        # shuffle the labels - note that we don't change the distribution of the class labels
        rn.shuffle(labels)
    return kmers_df, labels

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
                           
def load_kmer_cnts_for_t2d_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
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
    return kmers_df, labels

# Load all - MetaHIT has the option of using either IBD or Obesity as disease. This also has the option of
# excluding RA
def load_all_kmer_cnts_with_labels(exclude_ra=False, kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, metahit_obesity=False, shuffle_labels=False):
    kmers_df, labels = load_kmer_cnts_for_hmp_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels)
    # MetaHIT can have either IBD or Obesity as disease
    for (df, lbls) in [ load_kmer_cnts_for_metahit_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels) if not metahit_obesity else
                        load_metahit_with_obesity_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels),
                        load_kmer_cnts_for_t2d_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels),
                        load_kmer_cnts_for_ra_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels)
                        if not exclude_ra else (pd.DataFrame(), []) ]:
        # Matrix append
        kmers_df = kmers_df.append(df)
        # This is Python list merge
        labels += lbls
    if shuffle_labels:
        # shuffle the labels - note that we don't change the distribution of the class labels
        rn.shuffle(labels)
    return kmers_df, labels

# Load HMP and MetaHIT only
def load_hmp_metahit_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    kmers_df, labels = load_kmer_cnts_for_hmp_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels)
    for (df, lbls) in [ load_kmer_cnts_for_metahit_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels) ]:
        # Matrix append
        kmers_df = kmers_df.append(df)
        # This is Python list merge
        labels += lbls
    if shuffle_labels:
        # shuffle the labels - note that we don't change the distribution of the class labels
        rn.shuffle(labels)
    return kmers_df, labels

# Load RA and T2D only
def load_ra_t2d_with_labels(kmer_cnts_dir = '../data_generated', data_dir='../data', filter=True, load_1_only=False, shuffle_labels=False):
    kmers_df, labels = load_kmer_cnts_for_ra_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only, shuffle_labels)
    for (df, lbls) in [ load_kmer_cnts_for_t2d_with_labels(kmer_cnts_dir, data_dir, filter, load_1_only=load_1_only, shuffle_labels=shuffle_labels), ]:
        # Matrix append
        kmers_df = kmers_df.append(df)
        # This is Python list merge
        labels += lbls
    if shuffle_labels:
        # shuffle the labels - note that we don't change the distribution of the class labels
        rn.shuffle(labels)
    return kmers_df, labels

if __name__ == '__main__':
    # will tell you how much memory etc.

    for ((df, lbls), name) in [ (load_kmer_cnts_for_hmp_with_labels(), 'HMP'),
                                (load_kmer_cnts_for_metahit_with_labels(), 'MetaHIT'),
                                (load_kmer_cnts_for_t2d_with_labels(), 'T2D'),
                                (load_kmer_cnts_for_ra_with_labels(), 'RA') ]:
        

        print(name + ": total count - " + str( len(df.values)) + ", num of diseased - " + str([ lbls[i][0] for i in range(len(lbls)) ].count('1')))

    # kmers_df, labels = load_all_kmer_cnts_with_labels()
    # print("All data dims: " + str(kmers_df.values.shape))
    # kmers_df.info()
    # print("labels: " + str(len(labels)))
    # print(str(labels))

    # kmers_df, labels = load_all_kmer_cnts_with_labels(exclude_ra=True)
    # print("All data dims: " + str(kmers_df.values.shape))
    # kmers_df.info()
    # print("labels: " + str(len(labels)))
    # print(str(labels))

    # kmers_df = load_kmer_cnts_for_hmp(shuffle=True)
    # kmers_df.info()
    # print()
    # print("Shape")
    # print(kmers_df.values.shape)
    # print(kmers_df.values)
    # for r in kmers_df.values:
    #     print(np.sum(r))
    #     for i in range(len(r)):
    #         if (r[i] != 0.0):
    #             print('({}, {})'.format(all_kmers_caps[i], r[i]))


