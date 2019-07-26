import os.path
import subprocess
import argparse

# grep adapters from list of adapters that Sharon sent

def load_adapters():
    adapters=[]
    inFN=os.path.expanduser('~/deep_learning_microbiome/adapters/adaptors_list.fa')
    adapters=read_fasta(inFN, adapters)

    inFN=os.path.expanduser('~/deep_learning_microbiome/adapters/adapter_nexteraR1.fa')
    adapters=read_fasta(inFN, adapters)

    inFN=os.path.expanduser('~/deep_learning_microbiome/adapters/adapter_nexteraR2.fa')
    adapters=read_fasta(inFN, adapters)

    inFN=os.path.expanduser('~/deep_learning_microbiome/adapters/adapter_truseqR1.fa')
    adapters=read_fasta(inFN, adapters)
    
    inFN=os.path.expanduser('~/deep_learning_microbiome/adapters/adapter_truseqR2.fa')
    adapters=read_fasta(inFN, adapters)
    
    return adapters

#############

def read_fasta(inFN, adapters):
    inFile=open(inFN, 'r')
    for line in inFile:
        if line[0] != '>':
            adapters.append(line.strip())
    inFile.close()

    return adapters
            
#########

def sample_paths():
    paths={}

    # Karlsson
    dir=os.path.expanduser('~/shattuck/metagenomic_fastq_files/Karlsson_2013/fastq_files')
    list_of_samples=os.path.expanduser('~/shattuck/metagenomic_fastq_files/Karlsson_2013/PRJEB1786_run_accessions_only.txt')
    paths['Karlsson_2013']=[dir, list_of_samples]


    #RA:
    dir='/pollard/shattuck0/snayfach/metagenomes/RheumatoidArthritis/fastq'
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/RheumatoidArthritis_samples.txt')
    paths['RA']=[dir, list_of_samples]


    #Qin:
    dir="/pollard/shattuck0/snayfach/metagenomes/T2D/fastq"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/Qin_et_al_samples.txt')
    paths['Qin_et_al']=[dir, list_of_samples]

    #Zeller:
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/%s/combined_fastq_files"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/data/metadata/Zeller_2014_ids.txt')
    paths['Zeller_2014']=[dir, list_of_samples]

    # Feng:
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Feng_2015/combined_fastq_files"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/data/metadata/Feng_CRC_samples_only.txt')
    paths['Feng']=[dir, list_of_samples]

    # Liver Cirrhosis
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/%s/combined_fastq_files"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/data/metadata/LiverCirrhosis_ids.txt')
    paths['LiverCirrhosis']=[dir, list_of_samples]

    #MetaHIT
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/MetaHIT/combined_fastq_file"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/data/metadata/MetaHIT_samples_only.txt')
    paths['MetaHIT']=[dir, list_of_samples]

    #HMP
    dir="/pollard/data/metagenomes/HMP1-II/merged_sample_replicates"
    list_of_samples=os.path.expanduser('~/deep_learning_microbiome/data/metadata/HMP1-2_samples_only.txt')
    paths['HMP1-2']=[dir, list_of_samples]

    #Twins
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/twins/fastq_files"
    list_of_samples=os.path.expanduser('~/shattuck/metagenomic_fastq_files/twins/PRJEB9576_run_accessions_only.txt')
    paths['Twins']=[dir, list_of_samples]

    #IGC
    dir="/pollard/shattuck0/snayfach/metagenomes/IGC/fastq"
    list_of_samples=os.path.expanduser('~/shattuck/metagenomic_fastq_files/IGC/PRJEB5224_accessions_only.txt')
    paths['IGC']=[dir, list_of_samples]

    #Peru
    dir="/pollard/shattuck0/snayfach/metagenomes/Peru/fastq"
    list_of_samples=os.path.expanduser('~/shattuck/metagenomic_fastq_files/Peru/PRJNA268964_run_accession_only.txt')
    paths['Peru']=[dir, list_of_samples]

    #Nielsen
    dir="/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Nielsen_2014/combined_fastq_files"
    list_of_samples=os.path.expanduser('/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Nielsen_2014/PRJEB1220_sample_ids.txt')
    paths['Nielsen'] = [dir, list_of_samples]
    
    #LeChatelier
    dir=os.path.expanduser("~/shattuck/metagenomic_fastq_files/LeChatelier/combined_fastq_files")
    list_of_samples=os.path.expanduser("~/shattuck/metagenomic_fastq_files/LeChatelier/LeChatelier_sample_ids.txt")
    paths['LeChatelier'] = [dir, list_of_samples]

    # Fiji
    dir="/pollard/shattuck0/snayfach/metagenomes/FIJI/fastq"
    list_of_samples=os.path.expanduser("~/shattuck/metagenomic_fastq_files/FIJI/PRJNA217052_stool_accessions_only.txt")
    paths['Fiji'] = [dir, list_of_samples]

    # Backhed
    dir='/pollard/shattuck0/snayfach/metagenomes/Backhed/fastq'
    list_of_samples=os.path.expanduser("~/shattuck/metagenomic_fastq_files/Backhed/Backhed_mothers_only_accessions.txt")
    paths['Backhed']=[dir, list_of_samples]

    # Hadza
    dir='/pollard/shattuck0/snayfach/metagenomes/Hadza/fastq'
    list_of_samples=os.path.expanduser("~/shattuck/metagenomic_fastq_files/Hadza/sample_ids.txt")
    paths['Hadza']=[dir, list_of_samples]

    # Mongolian
    dir='/pollard/shattuck0/ngarud/metagenomic_fastq_files/Mongolian/fastq_files'
    list_of_samples='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Mongolian/PRJNA328899_run_accessions_only.txt'
    paths['Mongolian']=[dir, list_of_samples]

    # Raymond
    dir='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Raymond_2016/combined_fastq_files'
    list_of_samples='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Raymond_2016/day0_patient_ids.txt'
    paths['Raymond']=[dir, list_of_samples]
    
    # Yassour
    dir='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Yassour_2018/fastq_files'
    list_of_samples='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Yassour_2018/Mother_birth_accessions_only.txt'
    paths['Yassour']=[dir, list_of_samples]

    # Ferretti
    dir='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Ferretti_2018/fastq_files'
    list_of_samples='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Ferretti_2018/PRJNA352475_Mothers_Fecal_accessions.txt'
    paths['Ferretti']=[dir, list_of_samples]

    return paths


def grep_adapters(dataset, adapters, paths):

    outFile=open('/pollard/home/ngarud/deep_learning_microbiome/analysis/adapter_count/%s_adapter_count.txt' %dataset ,'w')

    outFile.write('dataset\tsample\tcountR1\tcountR2\tadapter\n')
    
    [dir, list_of_samples] = paths[dataset]
   
    
    file=open(list_of_samples,'r')
    count = 0
    for sample in file:
      if count <=10:
        sample=sample.strip()
        for adapter in adapters:
            cmd="zcat %s/%s_1.fastq.gz | grep %s | wc -l" %(dir, sample, adapter)
            output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            count_R1=output.decode().strip()

            cmd="zcat %s/%s_2.fastq.gz | grep %s | wc -l" %(dir, sample, adapter)
            output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            count_R2=output.decode().strip()

            outFile.write(dataset + '\t' + sample + '\t' + str(count_R1) + '\t' + str(count_R2) + '\t' + adapter + '\n')
            #print(dataset + '\t' + sample + '\t' + str(count_R1) + '\t' + str(count_R2) + '\t' + adapter + '\n')
        count +=1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= "Program to grep adapters from datasets")
    parser.add_argument('-dataset_in', type = str, default = 'MetaHIT', help = "Dataset")
    arg_vals = parser.parse_args()
    dataset = arg_vals.dataset_in

    # read in adapters
    adapters=load_adapters()

    # read in paths for the data sets and lists of samples
    paths=sample_paths()

    # grep the adapters for the sample of interest
    grep_adapters(dataset, adapters, paths)



