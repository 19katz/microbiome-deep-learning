import os
import gzip
import numpy

filename='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/IGC/PRJEB5224.txt'

    
# get mapping between run_accession and the sample_alias
metadata= {}
file = open(filename,"r")
file.readline() # header

for line in file:
    items = line.strip('\n').split("\t")
    run_accession = items[4]
    sample_id=items[5]
    if sample_id not in metadata:
        metadata[sample_id]=[]
    metadata[sample_id].append(run_accession)

file.close()
    
num_lines_exp={10:524800, 8:32896, 7:8192, 6:2080, 5:512}

for person in metadata:

    for kmer_size in [5, 6, 7, 8, 10]:
        
        combined_counts=numpy.zeros(num_lines_exp[kmer_size])

        os.system("rm /pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/IGC/%s_%smer.gz" %(kmer_size, person, kmer_size))

        run_accessions=metadata[person]

        for run_accession in run_accessions:
            print run_accession
            file=gzip.open('/pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/IGC/%s_%smer.gz' %(kmer_size, run_accession, kmer_size), 'rb')
            counts=[]
            for line in file:
                counts.append(int(line.strip()[1:]))
                if len(counts) == num_lines_exp[kmer_size]:
                    counts=numpy.asarray(counts)
                    combined_counts += counts

        outFN="/pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/IGC/%s_%smer.gz" %(kmer_size, person, kmer_size)
        outFile=gzip.open(outFN,'wb')
        for i in range(0, len(combined_counts)):
            outFile.write('>' +str(int(combined_counts[i])) +'\n')

