In order to run the two kmerizing files, all the data that needs to be processed should first be placed in the "data" folder in deep_learning_microbiome. 

A data_generated folder should be created at the same level as the data folder. This will store all of the count files generated through processing. 

parse_kmers is used to parse kmers of one size that's small (this size can be specified/changed in the file). As this one generates the kmer names in memory, a large kmer size shouldn't be used. For example, if you want to kmerize out 33-mers, using this program would need 4^33 keys just for the dictionary. 

parse_multiple_kmers is used to parse kmers of multiple sizes and/or large sizes (these sizes can also be specified/changed in the file). 

For both parse_kmers and parse_multiple_kmers, the number of threads (n_threads) should be set to however many cores there are on the machine being used. 

To set up the input data directory, I used symlinks to where the large fastq.gz files are stored. This way, I avoid copying these large files and also easily control which files are processed by deleting or creating symlinks.
