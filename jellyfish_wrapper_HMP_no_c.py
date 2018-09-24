import os.path
import subprocess

# For each batch, write a jellyfish command

def write_command(batch, kmer_size):
    command="while read file; do\n"
    command += "echo $file\n"
    command += "dir=/pollard/home/ngarud/BenNanditaProject/MIDAS_intermediate_files_hmp/joined_fastq_files_hmp_combine_tech_reps/\n"
    command += "jellyfish count <(zcat ${dir}/${file}_1.fastq.gz) <(zcat ${dir}/${file}_2.fastq.gz) /dev/fd/0 -m %s -s 100M -t 2 -C -F 2 -o ~/deep_learning_microbiome/data/%smers_jf/HMP/${file}_%smer.jf \n" %(kmer_size,kmer_size, kmer_size) 
    command += "jellyfish dump ~/deep_learning_microbiome/data/%smers_jf/HMP/${file}_%smer.jf | grep '>' | gzip > ~/deep_learning_microbiome/data/%smers_jf/HMP/${file}_%smer.gz\n" %(kmer_size,kmer_size, kmer_size, kmer_size)
    command += "done <  ~/deep_learning_microbiome/tmp_intermediate_files/HMP_batch_" + str(batch) + ".txt\n" 

    bash_script=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + 'HMP_batch_' + str(batch) + '_bash_script.sh'
    outFile=open(bash_script,'w')
    outFile.write(command)
    outFile.close()

    return bash_script

#########
# main  #
#########

kmer_size=8
number_of_processes=40

list_of_samples=os.path.expanduser('/pollard/home/ngarud/BenNanditaProject/MIDAS_intermediate_files_hmp/HMP_samples_314_no_c.txt')
file=open(list_of_samples,'r')

counter =0
batch = 1
list=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + 'HMP_batch_' + str(batch) + '.txt'  
outFile=open(list,'w')


command_file_inFN=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/all_commands_HMP_%smers.sh') % kmer_size
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
        list=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/') + 'HMP_batch_' + str(batch) + '.txt'  
        outFile=open(list,'w')
        outFile.write(sample)

# last batch        
outFile.close()
bash_script=write_command(batch, kmer_size)
# execute command
command='nohup nice bash %s &' %(bash_script)
command_file_inFile.write(command +'\n')
#subprocess.call(command.split())


