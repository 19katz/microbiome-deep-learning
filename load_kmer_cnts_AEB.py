import random as rn
import pandas as pd
import numpy as np
from itertools import product
from glob import glob
import re
import os.path
from os.path import basename
import pickle

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
def load_kmer_cnts_for_hmp(shuffle=False, kmer_cnts_dir = '/pollard/home/ngarud/deep_learning_microbiome/data/5mer_data_katherine', filter=True, load_1_only=False):
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

    #Add in kmer ids in a header row
    kmer_size = 5
    all_kmers = [''.join(_) for _ in product(['a', 'c', 'g', 't'], repeat = kmer_size)]
    kmers_df.columns = [all_kmers]

    #Add in file names in a header column
    no_path_files = [os.path.basename(x) for x in files]
    kmers_df.index = no_path_files

    kmers_df.to_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/loaded_katherine_5mers.pickle')
    return kmers_df

###To check ouput
#print(pd.read_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/loaded_katherine_5mers.pickle'))
#print(load_kmer_cnts_for_hmp())

###just the labels 174 X 1024
#def load_kmer_cnts_for_hmp_with_labels(kmer_cnts_dir = '/pollard/home/ngarud/deep_learning_microbiome/data/5mer_data_katherine', filter=True, load_1_only=False, shuffle_labels=False):
#    files = []
#    for id in hmp_no_timedup_ids:
#        if (not filter) or not (id in hmp_ids_to_be_filtered):
#            # Only use the _1 file names because the _2 ones (if exist) will be automatically loaded and merged with _1 file counts
#            files.append(kmer_cnts_dir + '/' + id + "_1.fastq.gz.5mer.rawcnt")
#    kmers_df = load_kmer_cnts_from_files(files, load_1_only=load_1_only)
#    # For HMP, shuffling labels or not doesn't matter because everyone's labels are the same -- healthy, US, and NA
#    return kmers_df, [ ['0', 'North America', None, 'United States', 'HMP', 'HMP', 'Healthy'] for i in range(len(kmers_df.values)) ]

#print(load_kmer_cnts_for_hmp_with_labels())
