# !/bin/bash

folder_path=$1
save_folder_path=$2
num_workers=$3

echo "folder_path: $folder_path"
echo "save_folder_path: $save_folder_path"
echo "num_workers: $num_workers"

# list all .mat file in &folder_path
find "$folder_path" -name "*.mat" > mat_files.txt

# run make_masks matlab for each line of mat_files.txt
counter=0
while read mat_file; do
    counter=$((counter+1))
    if [ $counter -eq $num_workers ]; then
        wait
        echo 'waiting!!!'
        counter=0
    fi
    file_path=$mat_file
    dataset_name=$(basename $(dirname "$mat_file"))
    echo --- $dataset_name
    python3 -u main.py make_mask_soma -fp "$file_path" -dn "$dataset_name" -sp "$save_folder_path" &
done < mat_files.txt

rm mat_files.txt
wait