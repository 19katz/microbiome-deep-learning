input_folder=./*.jf

for file in $input_folder;
do
mover_name=$(basename $file)
jellyfish dump $file -c > $mover_name.csv
done
