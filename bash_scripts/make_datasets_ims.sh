# !/bin/bash 
folder_path=$1
path2ims_reader=$2
savepath=$3

# find any ims files in the folder and subfolders
find "$folder_path" -name "*.ims" > ims_files.txt

# Counter
counter=0
# make ims_file reverse sorted
sort -r ims_files.txt -o ims_files.txt
# random shuffling if ims_files
shuf ims_files.txt -o ims_files.txt

# run make_dataset matlab function for each ims file
while read ims_file; do
    counter=$((counter+1))
    echo $ims_file
    # /usr/local/MATLAB/R2023a/bin/matlab -nodisplay -nosplash -nodesktop -r "make_dataset_ims('$path2ims_reader', '$ims_file', '$savepath'); exit;"
    matlab -nodisplay -nosplash -nodesktop -r "make_dataset_ims('$path2ims_reader', '$ims_file', '$savepath'); exit;" &
    # break the loop
    # if counter mod 10 is zero
    if [ $((counter % 3)) -eq 0 ]; then
        wait
    fi
done < ims_files.txt
wait

rm ims_files.txt