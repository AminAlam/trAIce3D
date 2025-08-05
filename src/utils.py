import numpy as np
from tqdm import tqdm
import torch

def reconstruct_image(imgs, metadata):
    img_size = metadata['img_size']
    window_size_x = metadata['window_size_x']
    window_size_y = metadata['window_size_y']
    window_size_z = metadata['window_size_z']
    overlap_x = metadata['overlap_x']
    overlap_y = metadata['overlap_y']
    overlap_z = metadata['overlap_z']
    img_all = np.zeros(img_size, dtype=np.uint8)
    print(img_size, int(img_size[0]/(window_size_x-overlap_x)))
    counter = 0
    for i_counter in tqdm(range(0, int(img_size[0]/(window_size_x-overlap_x))+1)):
        for j_counter in range(0, int(img_size[1]/(window_size_y-overlap_y))+1):
            for k_counter in range(0, int(img_size[2]/(window_size_z-overlap_z))+1):
                start_x = i_counter*(window_size_x-overlap_x)
                start_y = j_counter*(window_size_y-overlap_y)
                start_z = k_counter*(window_size_z-overlap_z)
                end_x = start_x+window_size_x
                end_y = start_y+window_size_y
                end_z = start_z+window_size_z

                if end_x > img_size[0]:
                    end_x = img_size[0]
                    start_x = end_x - window_size_x
                if end_y > img_size[1]:
                    end_y = img_size[1]
                    start_y = end_y - window_size_y
                
                if end_z > img_size[2]:
                    end_z = img_size[2]
                    start_z = end_z - window_size_z
                img_all[start_x:end_x, start_y:end_y, start_z:end_z] = imgs[counter, : , :, :]
                counter += 1
    return img_all

def reconstruct_images_no_overlap(imgs):
    num_imgs = imgs.shape[0]
    num_rows = int(np.sqrt(num_imgs))
    num_cols = int(np.ceil(num_imgs/num_rows))

    img_size = imgs.shape[1:]
    img_all = np.zeros((num_rows*img_size[0], num_cols*img_size[1], img_size[2]), dtype=np.float32)

    for row_counter in range(num_rows):
        for col_counter in range(num_cols):
            img_counter = row_counter*num_cols + col_counter
            if img_counter >= num_imgs:
                break

            start_x = row_counter*img_size[0]
            start_y = col_counter*img_size[1]

            end_x = start_x + img_size[0]
            end_y = start_y + img_size[1]
            img_all[start_x:end_x, start_y:end_y, :] = imgs[img_counter, :, :, :]
    
    return img_all


def convert_soma_and_traces_to_image(cells, img, convert_x, convert_y, convert_z, SizeX, SizeY, SizeZ):
    mask_soma = np.zeros(img.shape)
    mask_traces = np.zeros(img.shape)
    for cell in tqdm(cells.values()):
        traces = cell['traces']
        soma_pos = cell['soma_pos']
        x_soma = convert_x(soma_pos[0])
        y_soma = convert_y(soma_pos[1])
        z_soma = convert_z(soma_pos[2])
        mask_soma[x_soma, y_soma, z_soma] = 1
        kernel_size = 3
        # convolve soma with the kernel
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
        mask_soma = scipy.signal.convolve(mask_soma, kernel, mode='same')

        for trace_row in traces:
            x = convert_x(trace_row[0])
            y = convert_y(trace_row[1])
            z = convert_z(trace_row[2])
            if x >= SizeX:
                Warning(f'x out of range. Should be less than {SizeX}, but got {x}')
                x = SizeX-1
            if y >= SizeY:
                Warning(f'y out of range. Should be less than {SizeY}, but got {y}')
                y = SizeY-1
            if z >= SizeZ:
                Warning(f'z out of range. Should be less than {SizeZ}, but got {z}')
                z = SizeZ-1
            mask_traces[x, y, z] = 1
    
    return mask_soma, mask_traces


def prepare_image_for_plot(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    image = np.uint8(image*255)
    return image

def get_weights_of_the_model(model):
    weights = []
    for name, param in model.named_parameters():
        print(name)
        if 'weight' in name:
            weights.append(param)
    weights = torch.cat([w.view(-1) for w in weights])
    return weights

def normalize_img(x):
    x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
    return x