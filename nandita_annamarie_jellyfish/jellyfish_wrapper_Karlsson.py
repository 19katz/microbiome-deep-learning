import os.path
import subprocess

# For each batch, write a jellyfish command

def write_command(batch, kmer_size):
    name='Karlsson_2013'
    command="while read file; do\n"
    command += "echo $file\n"
    command += "dir=/pollard/home/ngarud/shattuck/metagenomic_fastq_files/%s/fastq_files/\n" %name
    command += "jellyfish count <(zcat ${dir}/${file}_1.fastq.gz) <(zcat ${dir}/${file}_2.fastq.gz) /dev/fd/0 -m %s -s 100M -t 2 -C -F 2 -o ~/deep_learning_microbiome/data/%smers_jf/%s/${file}_%smer.jf \n" %(kmer_size,kmer_size, name, kmer_size) 
    command += "jellyfish dump ~/deep_learning_microbiome/data/%smers_jf/%s/${file}_%smer.jf | grep '>' | gzip > ~/deep_learning_microbiome/data/%smers_jf/%s/${file}_%smer.gz\n" %(kmer_size,name, kmer_size, kmer_size, name, kmer_size)
    command += "done <  ~/deep_learning_microbiome/tmp_intermediate_files/%s_batch_"  %name + str(batch) + "_%smers.txt\n" %kmer_size 

    bash_script=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + '%s_batch_' %name + str(batch) + '_bash_script_%smers.sh' %kmer_size
    outFile=open(bash_script,'w')
    outFile.write(command)
    outFile.close()

    return bash_script

#########
# main  #
#########

kmer_size=10
number_of_processes=70
name='Karlsson_2013'

list_of_samples=os.path.expanduser('~/shattuck/metagenomic_fastq_files/Karlsson_2013/PRJEB1786_run_accessions_only.txt')

file=open(list_of_samples,'r')

counter =0
batch = 1
list=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + '%s_batch_' %name + str(batch) + '_%smers.txt' %kmer_size  
outFile=open(list,'w')


command_file_inFN=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/all_commands_%s_%smers.sh') %(name,kmer_size)
command_file_inFile=open(command_file_inFN,'w')

for sample in file:
    if counter <number_of_processes:
        outFile.write(sample) 
        counter +=1
    else:
        bash_script=write_command(batch, kmer_size)
        # execute command
        command='nohup nice bash %s &' %(bash_script)
        command_file_inFile.write(command +'\n')
        
        #update:
        counter=1
        batch +=1
        outFile.close()
        list=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + '%s_batch_' %name + str(batch) + '_%smers.txt' %kmer_size  
        outFile=open(list,'w')
        outFile.write(sample)

# last batch        
outFile.close()
bash_script=write_command(batch, kmer_size)
# execute command
command='nohup nice bash %s &' %(bash_script)
command_file_inFile.write(command +'\n')
#subprocess.call(command.split())


