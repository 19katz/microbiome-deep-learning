dataset=HMP1-2
for kmer_size in 5 6 8 10; do
    rm ~/deep_learning_microbiome/analysis/${dataset}_${kmer_size}_rerun.txt
    while read file; do
	file_path=~/deep_learning_microbiome/data/${kmer_size}mers_jf/${dataset}/${file}_${kmer_size}mer.gz
	count_zero=`zcat $file_path | grep '>0' | wc -l `
	if [ "$count_zero" -gt 0 ]; then
	    echo $file  >> ~/deep_learning_microbiome/analysis/${dataset}_${kmer_size}_rerun.txt
	fi
    done < ~/deep_learning_microbiome/data/metadata/HMP1-2_samples_only.txt
done
