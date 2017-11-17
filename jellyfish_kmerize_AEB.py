import subprocess
import gzip
from os import listdir
import os.path

jf_home = "/pollard/home/abustion/tools/jellyfish-2.2.6/bin/jellyfish"
source_dir = "/pollard/home/ngarud/BenNanditaProject/MIDAS_intermediate_files_hmp/joined_fastq_files_hmp_combi\
ne_tech_reps/"
dest_dir = "/pollard/home/abustion/deep_learning_microbiome/tmp_intermediate_files/"
results_dir = "/pollard/home/abustion/deep_learning_microbiome/data/jf_files/4mers_tech_reps/"

# Open each archive file
for src_name in listdir(source_dir):
    # Skip any not .gz
    if not src_name.endswith('.gz'):
        continue

    # Add full path to src_name
    src_name = source_dir + src_name
        
    # Setup our file names
    base = os.path.basename(src_name)[:-3] # get a name like 1231508394_1
    fastq_path = os.path.join(dest_dir, base)
    result_file = results_dir + base + '.jf'
    
    # If jf already exists, WE GOOD
    if os.path.isfile(result_file):
        continue
    
    # Open it with gzip, and write it out line by line!
    if not os.path.isfile(fastq_path):   
        print("Unzipping " + src_name + " to " + fastq_path)
        with gzip.open(src_name, 'rb') as infile:
            with open(fastq_path, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)
                
    # Now we have a file at dest_name, lets do stuff with it
    print("Doing work")
    subprocess.call(["echo", fastq_path])
    subprocess.call([jf_home, "count", "-m 4", "-s 1000", "-t 4", "-C", fastq_path, "-o" + result_file])
    
    # And finally, we want to clean up this file so we don't use up too much space
    print("Removing " + fastq_path)
    os.remove(fastq_path)
    
# We done
print("all done")
