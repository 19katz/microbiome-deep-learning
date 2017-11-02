###jellyfish count script###
# '-m' refers to kmer size
# '-s' refers to hash size
# '-t' refers to number of threads
# '-o' allows you to change output name
# '-C' folds reverse complements together into the same count

###
##Before using: specify,
#1. Input folder containing sequences to be kmerized
#2. Appropriate output file based on kmer size
#3. Choose kmer size (number after the '-m' parameter in jellyfish count)
###

input_folder='/pollard/home/ngarud/BenNanditaProject/MIDAS_intermediate_files_hmp/joined_fastq_files_hmp_combine_sample_reps/*.fastq.gz'
output_folder='/pollard/home/abustion/deep_learning_microbiome/data/jf_files/3mers'

#input_folder='/pollard/home/abustion/play/dummy_files/*fastq.gz'
#output_folder='/pollard/home/abustion/play/jellyfish_output_files'

for file in $input_folder;
do 
mover_name=$(basename $file)
echo $mover_name
zcat $file | jellyfish count /dev/fd/0 -m 3 -s 100M -t 2 -C -o $file.jf
mv $file.jf $output_folder/$mover_name.jf
done
