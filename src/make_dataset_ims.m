function make_dataset_ims(path2ims_reader, filepath, savepath)

addpath(path2ims_reader)

[~, file_name, ~] = fileparts(filepath);
% Check if dataset is created
if exist(fullfile(savepath, file_name, 'dataset.mat'), 'file')
    disp('Dataset already created')
    return
end


plot_bool = false;
save_dataset = true; 

% try
    % read text from filename with .txt instead of .ims at the end
    text_file = strrep(filepath, '.ims', '.txt');
    % check if text file exist
    if ~exist(text_file, 'file')
        channel_no = 1
        disp('No text file found, USING CHANNEL 1')
    else
        fileID = fopen(text_file,'r');
        formatSpec = '%f';
        channel_no = fscanf(fileID,formatSpec)
        fclose(fileID);
    end

    'start reading...'
    obj = ImarisReader(filepath, channel_no~=1);
    'read over...'
    try
        filaments = obj.Filaments.GID;
        filaments_all = FilamentsReader(filaments);
        idx = filaments_all.GetIndicesT;
        img = obj.DataSet.GetData;

        img = squeeze(img(:,:,:,channel_no));

        img = double(img);
        size(img)
        img = (img - min(img, [], 'all')) / (max(img, [], 'all') - min(img, [], 'all'));

        %% check if there is nan values
        if sum(isnan(img), 'all') > 0
            disp('There are NaN values in the image')
            return
        end

        dataset.img = img;
        dataset.metadata.ExtendMinX = obj.DataSet.ExtendMinX; % in um
        dataset.metadata.ExtendMinY = obj.DataSet.ExtendMinY; % in um
        dataset.metadata.ExtendMinZ = obj.DataSet.ExtendMinZ; % in um

        dataset.metadata.ExtendMaxX = obj.DataSet.ExtendMaxX; % in um
        dataset.metadata.ExtendMaxY = obj.DataSet.ExtendMaxY; % in um
        dataset.metadata.ExtendMaxZ = obj.DataSet.ExtendMaxZ; % in um

        dataset.metadata.SizeX = obj.DataSet.SizeX; % num voxels
        dataset.metadata.SizeY = obj.DataSet.SizeY; % num voxels
        dataset.metadata.SizeZ = obj.DataSet.SizeZ; % num voxels

        for i = 0:length(idx)-1
            pos = filaments_all.GetPositions(i);
            edges = filaments_all.GetEdges(i);
            radius = filaments_all.GetRadii(i);
            types = filaments_all.GetTypes(i);
            traces_label = zeros(length(pos), 1);
            label_no = 0;
            for j = 1:length(edges)
                p1 = edges(j, 1);
                p2 = edges(j, 2);
                if p2-p1 ~= 1
                    label_no = label_no+1;
                end
                traces_label(j) = label_no;
            end
            color_mat = zeros(length(pos), 3);
            dataset.cells.(sprintf("cell%d",i)).traces_pos = pos;
            dataset.cells.(sprintf("cell%d",i)).traces_edges = edges;
            dataset.cells.(sprintf("cell%d",i)).traces_radius = radius;
            dataset.cells.(sprintf("cell%d",i)).traces_types = types;
            dataset.cells.(sprintf("cell%d",i)).soma_pos = [pos(1, 1), pos(1, 2), pos(1, 3)];
            dataset.cells.(sprintf("cell%d",i)).traces_label = traces_label;
            if plot_bool
                color_now = randn([3,1]);
                for j = 1:length(edges)
                    p1 = edges(j, 1);
                    p2 = edges(j, 2);
                    if p2-p1 ~= 1
                        color_now = randn([3,1]);
                    end
                    color_mat(j, :) = color_now;
                end
                color_mat = uint8(color_mat/max(color_mat, [], 'all')*255);
                scatter3(pos(1,1), pos(1,2), pos(1,3), radius(1)*10, 'red', 'filled')
                hold on
                scatter3(pos(:, 1), pos(:, 2), pos(:, 3), radius, color_mat, 'filled');
                hold off
            end
        end
    catch ME
        disp('ERROR')
        % rename file
        movefile(filepath, fullfile(savepath, file_name, '.old_ims'))
    end

    if save_dataset
        [~, file_name, ~] = fileparts(filepath);
        
        save_path_this_file = fullfile(savepath, file_name);
        if ~ exist(save_path_this_file, 'dir')
            mkdir(save_path_this_file)
        end
        save_path_this_file = fullfile(savepath, file_name, 'dataset.mat');
        save(save_path_this_file, 'dataset', '-v7.3')
    end
    % catch ME
    %     disp('ERROR')
    % end

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