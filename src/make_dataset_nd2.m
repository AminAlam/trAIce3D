function make_dataset_nd2(path2nd2_reader, filepath, savepath)
    addpath(genpath(path2nd2_reader))
    
    plot_bool = false;
    save_dataset = true;

        [dataset.img, info] = load_nd2(filepath);
        % check if obj has filaments
        % img = adapthisteq3D(img);
        % img = medfilt3(img, 3);
        % img = imbinarize(img, 'adaptive').*255;
        % img = medfilt3(img, 9);
        size_img = size(dataset.img);
        min_img = min(dataset.img, [], 'all');
        max_img = max(dataset.img, [], 'all');

        win_size_x = 200;
        win_size_y = 200;
        win_size_z = 20;

        tic
        counter = 0;

        % size_img(1) % 17617
        % size_img(2) % 17618
        % size_img(3) % 51
        % ceil(size_img(1)/win_size_x)-1 % 88
        % ceil(size_img(2)/win_size_y)-1 % 88
        % ceil(size_img(3)/win_size_z)-1 % 2
 
        for i_counter = 0:ceil(size_img(1)/win_size_x)-1
            for j_counter = 0:ceil(size_img(2)/win_size_y)-1
                for k_counter = 0:ceil(size_img(3)/win_size_z)-1
                    start_x = i_counter*win_size_x+1;
                    start_y = j_counter*win_size_y+1;
                    start_z = k_counter*win_size_z+1;
                    end_x = start_x+win_size_x-1;
                    end_y = start_y+win_size_y-1;
                    end_z = start_z+win_size_z-1;

                    if end_x > size_img(1)
                        end_x = size_img(1);
                    end
    
                    if end_y > size_img(2)
                        end_y = size_img(2);
                    end
    
                    if end_z > size_img(3)
                        end_z = size_img(3);
                    end
                    % disp(start_x)
                    % disp(end_x)
                    % disp(start_y)
                    % disp(end_y)
                    % disp(start_z)
                    % disp(end_z)
                    % return
                    dataset.img(start_x:end_x, start_y:end_y, start_z:end_z) = (dataset.img(start_x:end_x, start_y:end_y, start_z:end_z)-min_img)/(max_img-min_img);
                    disp(counter/(floor(size_img(1)/win_size_x)*floor(size_img(2)/win_size_y)*floor(size_img(3)/win_size_z)))
                    toc
                    counter = counter+1;
                end
            end
        end

        dataset.img = uint8(dataset.img * 255);

        dataset.metadata.SizeX = info.ImageWidth; % num voxels
        dataset.metadata.SizeY = info.ImageHeight; % num voxels
        dataset.metadata.SizeZ = info.numImages; % num voxels

        if save_dataset
            [~, file_name, ~] = fileparts(filepath);
            
            save_path_this_file = fullfile(savepath, file_name);
            if ~ exist(save_path_this_file, 'dir')
                mkdir(save_path_this_file)
            end
            save_path_this_file = fullfile(savepath, file_name, 'dataset.mat');
            save(save_path_this_file, 'dataset', '-v7.3')
        end

    
    end
