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

kmer_size = 7
# All the kmer names
all_kmers_caps = [''.join(_) for _ in product(['A', 'C', 'G', 'T'], repeat = kmer_size)]

# Parse out the kmers from a gzipped fastq file and
# write the results to a file as a line of counts separated by tabs
# - the kmer counts in an output file are in the kmers' lexical order
# - the output file is named as <inputfile-name>.{k}mer.rawcnt
def get_kmers(output_dir, fn):
    print("Processing: " + fn)
    d = {}.fromkeys(all_kmers_caps, 0.0)

    # running through the sequences
    for sq in SeqIO.parse(gzip.open(fn, 'rt'), 'fastq'):
        sequence = str(sq.seq)
        # parsing out every kmer
        for i in range(len(sequence) - kmer_size + 1):
            kmer = sequence[i:i+kmer_size]
            try:
                d[kmer] += 1
            # in case there are N's in the sequence
            except KeyError:
                pass

    # sort counts in lexical kmer order
    values = np.asarray([d[k] for k in all_kmers_caps])
    result_file = '{}/{}.{}mer.rawcnt'.format(output_dir, ntpath.basename(fn), kmer_size)
    with open(result_file, 'w') as f:
        f.write("\t".join([str(n) for n in values]) + "\n")
    # return np.divide(values, np.sum(values))
    return values / np.sum(values)

# Given an input directory and file name pattern of gzipped fastq files,
# spawn the specified number of threads to parse out kmer counts in
# the input files. These counts are written to output files, one per input file.
def process_fastq_files(file_name_list, output_dir='../data_generated',
                       n_threads=6):
    pool = Pool(n_threads)
    part_f = partial(get_kmers, output_dir)
    kmer_cnts_list = pool.map(part_f, file_name_list)

    # create pandas dframe
    return pd.DataFrame(
        # stack the arrays vertically into a numpy array
        np.vstack(kmer_cnts_list),
        # naming all the columns with the kmer names in caps
        columns = all_kmers_caps,
    )

def process_fastq_data(file_pattern='*.fastq.gz', output_dir = '../data_generated',
                       input_dir = '../data', n_threads=6):
    return process_fastq_files(glob(input_dir + '/' + file_pattern), output_dir, n_threads)

if __name__ == '__main__':
    kmers_df = process_fastq_data()
    # will tell you how much memory etc.
    kmers_df.info()

    print("Kmer counts vector shape: " + str(kmers_df.values.shape))
    '''print("Kmer counts: " + str(kmers_df.values))
    for r in kmers_df.values:
        print("Row sum: " + str(np.sum(r)))
        for i in range(len(r)):
            if (r[i] != 0.0):
                print('({}, {})'.format(all_kmers_caps[i], r[i]))'''
