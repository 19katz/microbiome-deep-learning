input_folder=./*.jf
output_folder=./4mer_csv_files/

for file in $input_folder;
do
mover_name=$(basename $file)
jellyfish dump $file -c > $output_folder$mover_name.csv
done
