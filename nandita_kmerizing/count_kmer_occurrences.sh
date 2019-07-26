dir='/pollard/home/ngarud/shattuck/metagenomic_fastq_files/Karlsson_2013/fastq_files/'
rm ~/deep_learning_microbiome/analysis/importances_karlsson_RF_10mer_frequencies.txt
echo -e "accession\tkmer\tscore\tcount_f_1\tcount_f_2\tcount_r_1\tcount_r_2" > ~/deep_learning_microbiome/analysis/importances_karlsson_RF_10mer_frequencies.txt
while read line; do
    kmer=`echo $line | cut -f1 -d' '`
    score=`echo $line | cut -f2 -d' '`
    echo $kmer
    rev_kmer=`echo $kmer | rev | tr ATGC TACG`
    echo $rev_kmer
    while read accession; do
	echo $accession
	count_f_1=`zcat ${dir}/${accession}_1.fastq.gz | grep $kmer | wc -l`
	count_f_2=`zcat ${dir}/${accession}_2.fastq.gz | grep $kmer | wc -l`
	count_r_1=`zcat ${dir}/${accession}_1.fastq.gz | grep $rev_kmer | wc -l`
	count_r_2=`zcat ${dir}/${accession}_2.fastq.gz | grep $rev_kmer | wc -l`
	echo -e "$accession\t$kmer\t$score\t$count_f_1\t$count_f_2\t$count_r_1\t$count_r_2" >> ~/deep_learning_microbiome/analysis/importances_karlsson_RF_10mer_frequencies.txt
    done < ~/shattuck/metagenomic_fastq_files/Karlsson_2013/PRJEB1786_run_accessions_only.txt
done < ~/deep_learning_microbiome/analysis/importances_karlsson_linear_best_model.txt
