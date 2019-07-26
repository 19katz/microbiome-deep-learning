import os

# the purpose of this script is to rename the fastq files beacuse there is a discrepancy between their names and the metadata


# list of files without a dash:
filename='/pollard/home/ngarud/deep_learning_microbiome/data/metadata/LiverCirrhosis_ids.txt'

    
# get mapping between run_accession and the sample_alias
file = open(filename,"r")
file.readline() # header

for line in file:
    person=line.strip('\n')
    if '-' not in person:
        new_person= person[0:2] + '-' + person[2:len(person)] 
        print person + '\t' + new_person
        
        for kmer_size in [5, 6, 7, 8, 10]:
            os.system("mv /pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/LiverCirrhosis/%s_%smer.jf /pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/LiverCirrhosis/%s_%smer.jf" %(kmer_size, person, kmer_size, kmer_size, new_person, kmer_size))

            os.system("mv /pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/LiverCirrhosis/%s_%smer.gz /pollard/home/ngarud/deep_learning_microbiome/data/%smers_jf/LiverCirrhosis/%s_%smer.gz" %(kmer_size, person, kmer_size, kmer_size, new_person, kmer_size))


