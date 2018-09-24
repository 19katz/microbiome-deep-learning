import gzip
import sys
import os.path

inFN=sys.argv[1]
outFN=sys.argv[2]
kmer_size=int(sys.argv[3])

# first read in the data with the kmers into a dictionary

inFile=open(os.path.expanduser(inFN),'r')

kmer_dict={} # key=kmer, value=number

for line in inFile:
    if line[0] == '>':
        number=line
    else:
        kmer=line
        kmer_dict[kmer]=number

# read in the order of the kmers

inFile_order=gzip.open(os.path.expanduser("~/deep_learning_microbiome/data/%smers_jf/%smer_dictionary.gz" %(kmer_size, kmer_size)), 'rb')
 
outFile=gzip.open(os.path.expanduser(outFN),'wb')

for line in inFile_order:
    if line in kmer_dict:
        outFile.write(kmer_dict[line])
    else:
        outFile.write('>' + str(0) + '\n')
