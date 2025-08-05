function make_dataset_ims_fluer(path2ims_reader, filepath, savepath)

[~, file_name, ~] = fileparts(filepath);

save_path_this_file = fullfile(savepath, file_name, 'dataset.mat');
if ~ exist(save_path_this_file)
    mkdir(fullfile(savepath, file_name))
else
    disp('Dataset already exists')
    return
end

addpath(path2ims_reader)
addpath(genpath('./progress'))

plot_bool = false;
save_dataset = true;

% try
    obj = ImarisReader(filepath);
    % break_bool = check_filaments(obj.Filaments);

    % if break_bool
    %     return
    % end
    num_spots = length(obj.Spots);

    dataset.metadata.ExtendMinX = obj.DataSet.ExtendMinX; % in um
    dataset.metadata.ExtendMinY = obj.DataSet.ExtendMinY; % in um
    dataset.metadata.ExtendMinZ = obj.DataSet.ExtendMinZ; % in um

    dataset.metadata.ExtendMaxX = obj.DataSet.ExtendMaxX; % in um
    dataset.metadata.ExtendMaxY = obj.DataSet.ExtendMaxY; % in um
    dataset.metadata.ExtendMaxZ = obj.DataSet.ExtendMaxZ; % in um

    dataset.metadata.SizeX = obj.DataSet.SizeX; % num voxels
    dataset.metadata.SizeY = obj.DataSet.SizeY; % num voxels
    dataset.metadata.SizeZ = obj.DataSet.SizeZ; % num voxels

    all_pos = [];
    all_radius = [];
    for i=progress(1:num_spots)
        filaments = obj.Spots(i).GID;
        filaments_all = SpotsReader(filaments);
        idx = filaments_all.GetIndicesT;
        all_pos = cat(1,all_pos,filaments_all.GetPositions());
        all_radius = cat(1,all_radius,filaments_all.GetRadii());
    end
    size(all_pos)
    size(all_radius)
    
    for i = progress(1:size(all_pos,1))
        pos = all_pos(i,:);
        radius = all_radius(i,:);

        dataset.cells.(sprintf("cell%d",i)).traces_pos = pos;
        dataset.cells.(sprintf("cell%d",i)).traces_radius = radius;
    end

    disp('Loading Image...')
    img = obj.DataSet.GetData;
    img = squeeze(img(:,:,:,1));
    img = double(img);
    size(img)
    img = (img - min(img, [], 'all')) / (max(img, [], 'all') - min(img, [], 'all'));
    img = img * 255;
    img = uint8(img);
    dataset.img = img;

    disp('Start Saving...')
    save(save_path_this_file, 'dataset', '-v7.3')
end

function break_bool = check_filaments(filaments_obj)
    % check if obj has filaments
    if isempty(filaments_obj)
        disp('No filaments found in this file')
        break_bool = true;
        return 
    end
    % num_filaments_objects = numel(filaments_obj); 
    % for i = 1:num_filaments_objects
    %     filaments = FilamentsReader(filaments_obj(i).GID);
    %     disp(i)
    %     disp(filaments.Name)
    % end
    break_bool = false;
end