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


kmer_sizes = [7, 9]

def get_kmers(output_dir, fn):
    print("Processing: " + fn)
    dicts = { size : {} for size in kmer_sizes }

    # running through the sequences
    for sq in SeqIO.parse(gzip.open(fn, 'rt'), 'fastq'):
        sequence = str(sq.seq)
        leng = len(sequence)
        # parsing out every kmer
        for size in kmer_sizes:
            d = dicts[size]
            for i in range(leng - size + 1):
                kmer = sequence[i:i+size]
                try:
                    d[kmer] += 1
                except KeyError:
                    d[kmer] = 1

    for size in kmer_sizes:
        d = dicts[size]
        result_file = '{}/{}.{}mer_by_name'.format(output_dir, ntpath.basename(fn), size)
        with open(result_file, 'w') as f:
            for k, v in d.items():
                f.write(k + '\t' + str(v) + "\n")

def process_fastq_files(file_name_list, output_dir = '../data_generated',
                    n_threads=6):
    pool = Pool(n_threads)
    part_f = partial(get_kmers, output_dir)
    pool.map(part_f, file_name_list)

def process_fastq_data(file_pattern='*.fastq.gz', output_dir = '../data_generated',
                       input_dir = '../data', n_threads=6):
    process_fastq_files(glob(input_dir + '/' + file_pattern), output_dir, input_dir, n_threads)



if __name__ == '__main__':
    process_fastq_data()
