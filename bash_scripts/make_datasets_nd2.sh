# !/bin/bash 
folder_path=$1
path2ims_reader=$2
savepath=$3

# find any ims files in the folder and subfolders
find "$folder_path" -name "*.ims" > ims_files.txt

# run make_dataset matlab function for each ims file
while read ims_file; do
    echo $ims_file
    # /usr/local/MATLAB/R2023a/bin/matlab -nodisplay -nosplash -nodesktop -r "make_dataset_ims('$path2ims_reader', '$ims_file', '$savepath'); exit;"
    /usr/local/MATLAB/R2023a/bin/matlab -nodisplay -nosplash -nodesktop -r "make_dataset_nd2('$path2ims_reader', '$ims_file', '$savepath'); exit;"
done < ims_files.txt

rm ims_files.txt