# !/bin/bash

folder_path=$1
save_folder_path=$2
repeat_num=$3
postfix=$4

# find all folder paths in the directory and subdirectories
# find "$folder_path" -type d -maxdepth 1 -mindepth 1 > folders.txt

# run make_dataset matlab function for each ims file
# while read data_path; do
#     echo $data_path
python3 -u main.py augment_datasets_in_a_folder -fp "$folder_path" -rn "$repeat_num" -sp "$save_folder_path" -pf "$postfix"
# done < folders.txt

# wait
# rm folders.txt