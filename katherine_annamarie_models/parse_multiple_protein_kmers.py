#!/usr/bin/env python3

import gzip
import ntpath

from Bio import SeqIO, Seq
from glob import glob

from functools import partial
from multiprocessing import Pool


def get_kmers(kmer_sizes, output_dir, fn):
    print("Processing: " + fn)
    dicts = { size : {} for size in kmer_sizes }

    # running through the sequences
    for sq in SeqIO.parse(gzip.open(fn, 'rt'), 'fastq'):
        # print("  Original DNA seq: '" + str(sq.seq) + "'")
        sequences = str(sq.seq).split('N')
        # print("    After removing Ns: '" + ', '.join(sequences) + "'")
        
        for sequ in sequences:
            if len(sequ) < 3:
                continue

            for sequence in (sequ, sequ[::-1]):
                for i in [0, 1, 2]:
                    buf = sequence[i:]
                    protein_seq = Seq.translate(buf)
                    # print("      Translated shift '" + buf + "' to: " + protein_seq)
        
                    protein_seqs = protein_seq.split('*')
                    for seq in protein_seqs:
                        leng = len(seq)
                        if leng == 0:
                            continue
                
                        # parsing out every kmer
                        for size in kmer_sizes:
                            d = dicts[size]
                            # Use while instead of for i in range() to avoid creating
                            # lists too many times.
                            i = 0
                            while i < leng - size + 1:
                                kmer = seq[i:i+size]
                                # print("        Adding protein {} kmer: '{}'".format(size, kmer))
                                try:
                                    d[kmer] += 1
                                except KeyError:
                                    d[kmer] = 1
                                i += 1

    for size in kmer_sizes:
        d = dicts[size]
        result_file = '{}/{}.protein_{}mer_by_name'.format(output_dir, ntpath.basename(fn), size)
        with open(result_file, 'w') as f:
            for k, v in d.items():
                f.write(k + '\t' + str(v) + "\n")

def process_fastq_files(kmer_sizes, file_name_list, output_dir = '../data_generated',
                        n_threads=6):
    pool = Pool(n_threads)
    part_f = partial(get_kmers, kmer_sizes, output_dir)
    pool.map(part_f, file_name_list)

def process_fastq_data(kmer_sizes, file_pattern='*.fastq.gz',
                       output_dir = '../data_generated', input_dir = '../data', n_threads=6):
    process_fastq_files(kmer_sizes, glob(input_dir + '/' + file_pattern), output_dir, n_threads)



if __name__ == '__main__':
    # for kmer_sizes in ([5, 7, 9, 11],):
    #     process_fastq_data(kmer_sizes, file_pattern='tt.fastq.gz',
    #                        output_dir = 'tmp', input_dir = 'tmp', n_threads=1)
    process_fastq_data([5, 7, 9, 11], n_threads=6)
