dataset=IGC
dir=/pollard/shattuck0/snayfach/metagenomes/IGC/fastq


for kmer_size in 5 6 7 8 10; do
    echo $kmer_size

    #rm -R ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}
    mkdir ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}

    # get the num lines expected for the given kmer_size
    
    if [ $kmer_size = 10 ]
    then
	num_lines_exp=524800
    elif [ $kmer_size = 8 ]
    then
	num_lines_exp=32896
    elif [ $kmer_size = 7 ]
    then
	num_lines_exp=8192
    elif [ $kmer_size = 6 ]
    then 
	num_lines_exp=2080
    elif [ $kmer_size = 5 ]
    then
	num_lines_exp=512
    fi

    # iterate through each file: 

    while read file; do 
	echo $file

    jellyfish count <(zcat ${dir}/${file}_1.fastq.gz) <(zcat ${dir}/${file}_2.fastq.gz) /dev/fd/0 -m ${kmer_size} -s 100M -t 2 -C -F 2 -o ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.jf 

    jellyfish dump ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.jf | grep '>' | gzip > ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.gz

    # check if jellyfish ran correctly

    num_lines=`zcat  ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.gz | wc -l | cut -f1 -d' '`

    if [ $num_lines = $num_lines_exp ]
    then
	echo 'file is ok...'
    else
	echo 'filling in zeros...'

    # rerun jellyfish
    jellyfish count <(zcat ${dir}/${file}_1.fastq.gz) <(zcat ${dir}/${file}_2.fastq.gz) /dev/fd/0 -m $kmer_size -s 100M -t 2 -C -F 2 -o ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.jf

    jellyfish dump ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.jf > ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_tmpOut.txt

    # run a script to find the missing kmers and fill in a zero
    ~/miniconda3/bin/python3 ~/deep_learning_microbiome/scripts/fill_in_zeros.py $file $kmer_size $dataset

    rm ~/deep_learning_microbiome/data/${kmer_size}mers_jf/${data_set}/${file}_tmpOut.txt

    fi

done <~/shattuck/metagenomic_fastq_files/IGC/accessions_single.txt

done


