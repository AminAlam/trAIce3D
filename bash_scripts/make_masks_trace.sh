# !/bin/bash

folder_path=$1
save_folder_path=$2
window_size_x=$3
window_size_y=$4
window_size_z=$5
num_instance=$6
steps=$7

# list all .mat file in &folder_path
find "$folder_path" -name "*.mat" > mat_files.txt
# Counter
shuf mat_files.txt -o mat_files.txt
counter=0

# run make_masks matlab for each line of mat_files.txt
while read mat_file; do
    counter=$((counter+1))
    file_path=$mat_file
    dataset_name=$(basename $(dirname $mat_file))
    echo --- $dataset_name
    python3 -u main.py make_mask_trace -fp "$file_path" -dn "$dataset_name" -sp "$save_folder_path" --window_size_x "$window_size_x" --window_size_y "$window_size_y" --window_size_z "$window_size_z" --no_random_shifts "$num_instance" &
    if [ $((counter % steps)) -eq 0 ]; then
        wait
    fi
done < mat_files.txt
wait
rm mat_files.txt

# # Function to decrement counter
# decrement_counter() {
#     counter=$((counter-1))
#     echo "Counter: $counter"
# }

# # run make_masks matlab for each line of mat_files.txt
# while read mat_file; do
#     counter=$((counter+1))
#     echo "Counter: $counter"
#     file_path=$mat_file
#     dataset_name=$(basename $(dirname $mat_file))
#     echo --- $dataset_name
#     python3 -u main.py make_mask_trace -fp "$file_path" -dn "$dataset_name" -sp "$save_folder_path" --window_size_x "$window_size_x" --window_size_y "$window_size_y" --window_size_z "$window_size_z" --no_random_shifts "$num_instance" &
#     pid=$!
#     wait 1
#     # decrement counter when pid is done
#     trap decrement_counter CHLD

#     if [ $((counter - steps)) -eq 0 ]; then
#         while [ $counter -gt 0 ]; do
#             sleep 10
#         done
#     fi
# done < mat_files.txt
# wait
# rm mat_files.txt