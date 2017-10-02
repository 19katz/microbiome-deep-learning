

import gzip
#the “as pd” and “as np” are to avoid overlap with built-in methods in python like max() and min()
import pandas as pd 
import numpy as np 
#looks like this is for operating system functionality on a Windows
##not sure why it was needed
import ntpath

#from BioPython, importing SeqIO (sequence input/output) 
from Bio import SeqIO
#glob module finds pathnames matching a specified pattern
from glob import glob
#looks like this is for iterator algebra. helps make loops more efficient.
##not sure where it was used
from itertools import product

#makes a new version of a function with one more arguments already filled in
from functools import partial
#is this for thread stuff?
from multiprocessing import Pool

kmer_sizes = [2]

def get_kmers(output_dir, fn):
    print("Processing: " + fn)
    count = 0
    dicts = {
        size : {} for size in kmer_sizes
    }

    # running through the sequences
    #gzip.open lets me open a compressed file in text mode and returns file object
    ##fn is the filename
    ##’rt’ is open for read, text mode (not binary)
    for sq in SeqIO.parse(gzip.open(fn, 'rt'), 'fastq'):
        #cutting up into pieces separated by N's
        sequences = str(sq.seq).split('N')
        for sequence in sequences:
            print("still going...")
            leng = len(sequence)
            count = count + 1
            print("processing sequence count of: " + str(count))
            if leng == 0:
                continue

            # parsing out every kmer
            for size in kmer_sizes:
                d = dicts[size]
                # Use while instead of for i in range() to avoid creating
                # lists too many times.
                i = 0
                while i < leng - size + 1:
                    kmer = sequence[i:i+size]
                    try:
                        #Scanning the frame of length (size) over by one until frame can't be accomodated by sequence length
                        d[kmer] += 1
                    except KeyError:
                        d[kmer] = 1
                    i += 1

    for size in kmer_sizes:
        d = dicts[size]
        result_file = '{}/{}.{}mer_by_name'.format(output_dir, ntpath.basename(fn), size)
        with open(result_file, 'w') as f:
            for k, v in d.items():
                f.write(k + '\t' + str(v) + "\n")

def process_fastq_files(file_name_list, output_dir = 'data_generated',
                    n_threads=6):
    pool = Pool(n_threads)
    part_f = partial(get_kmers, output_dir)
    pool.map(part_f, file_name_list)

def process_fastq_data(file_pattern='*.fastq.gz', output_dir = 'data_generated',
                       input_dir = 'data', n_threads=6):
    process_fastq_files(glob(input_dir + '/' + file_pattern), output_dir, n_threads)



if __name__ == '__main__':
    #process_fastq_data(output_dir = 'tmp', input_dir = 'tmp')
    process_fastq_data(n_threads=6)

process_fastq_data()
