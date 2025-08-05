import os
import click
import mat73
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import json
import scipy
import datasets
import utils
import tifffile as tiff
from skimage import exposure
from skimage.transform import rescale as skrescale
from copy import deepcopy
from math import ceil, floor
import torchio as tio
import imageio
from concurrent import futures 

import models
import torch

@click.group(chain=True)
def cli():
    pass

@cli.command('make_mask_soma')
@click.option('--file_path', '-fp', type=str, required=True)
@click.option('--dataset_name', '-dn', type=str, required=True)
@click.option('--save_path_dataset', '-sp', type=str, required=True)
@click.option('--window_size_x', '-wsx', type=int, required=True, default=256)
@click.option('--window_size_y', '-wsy', type=int, required=True, default=256)
@click.option('--window_size_z', '-wsz', type=int, required=True, default=32)
@click.option('--overlap_x', '-ox', type=int, required=True, default=32)
@click.option('--overlap_y', '-oy', type=int, required=True, default=32)
@click.option('--overlap_z', '-oz', type=int, required=True, default=4)
@click.option('--trace_size_z', '-tsxy', type=int, required=True, default=0)
@click.option('--kernel_size', '-ks', type=int, required=True, default=40)
@click.option('--target_x_px2mu', '-tx', type=float, required=True, default = 2.5)
@click.option('--target_y_px2mu', '-ty', type=float, required=True, default = 2.5)
@click.option('--target_z_px2mu', '-tz', type=float, required=True, default = 0.8)
def make_mask_soma(file_path, target_x_px2mu, target_y_px2mu, target_z_px2mu, dataset_name, save_path_dataset, window_size_x, window_size_y, window_size_z, overlap_x, overlap_y, overlap_z, trace_size_z, kernel_size):
    dataset = mat73.loadmat(file_path)
    dataset = dataset['dataset']
    img = dataset['img']

    cells = dataset['cells']
    ExtendMinX = dataset['metadata']['ExtendMinX']
    ExtendMinY = dataset['metadata']['ExtendMinY']
    ExtendMinZ = dataset['metadata']['ExtendMinZ']
    ExtendMaxX = dataset['metadata']['ExtendMaxX']
    ExtendMaxY = dataset['metadata']['ExtendMaxY']
    ExtendMaxZ = dataset['metadata']['ExtendMaxZ']
    SizeX = int(dataset['metadata']['SizeX'])
    SizeY = int(dataset['metadata']['SizeY'])
    SizeZ = int(dataset['metadata']['SizeZ'])
    origin_x_px2mu = SizeX/(ExtendMaxX-ExtendMinX)
    origin_y_px2mu = SizeY/(ExtendMaxY-ExtendMinY)
    origin_z_px2mu = SizeZ/(ExtendMaxZ-ExtendMinZ)
    # print(f"origin_x_px2mu: {origin_x_px2mu} - origin_y_px2mu: {origin_y_px2mu} - origin_z_px2mu: {origin_z_px2mu}")

    size_x_new = int(SizeX/origin_x_px2mu*target_x_px2mu)
    size_y_new = int(SizeY/origin_y_px2mu*target_y_px2mu)
    size_z_new = int(SizeZ/origin_z_px2mu*target_z_px2mu)

    ratio_x = size_x_new/SizeX
    ratio_y = size_y_new/SizeY
    ratio_z = size_z_new/SizeZ
    print(f"ratio_x: {ratio_x} - ratio_y: {ratio_y} - ratio_z: {ratio_z}")

    img = scipy.ndimage.zoom(img, (ratio_x, ratio_y, ratio_z))
    # rescale the img to have max = 1 and min = 0
    img = img - np.min(img)
    img = img/np.max(img)

    SizeX = img.shape[0]
    SizeY = img.shape[1]
    SizeZ = img.shape[2]
    print(f"Size: {img.shape}")

    convert_x = lambda x: int((x-ExtendMinX)/(ExtendMaxX-ExtendMinX)*SizeX)
    convert_y = lambda y: int((y-ExtendMinY)/(ExtendMaxY-ExtendMinY)*SizeY)
    convert_z = lambda z: int((z-ExtendMinZ)/(ExtendMaxZ-ExtendMinZ)*SizeZ)
    convert_radius_x = lambda r: ceil(r/(ExtendMaxX-ExtendMinX)*SizeX)
    convert_radius_y = lambda r: ceil(r/(ExtendMaxY-ExtendMinY)*SizeY)
    convert_radius_z = lambda r: ceil(r/(ExtendMaxZ-ExtendMinZ)*SizeZ)


    print(f'- Making 3d images from somas and traces of the cells ...\n Image size: {img.shape} - Number of cells: {len(cells)} - SizeX: {SizeX} - SizeY: {SizeY} - SizeZ: {SizeZ}')
    if len(img.shape) > 3:
        return print('Error: Image has more than 3 dimensions')
    mask_soma = np.zeros(img.shape)
    mask_traces = np.zeros(img.shape)
    
    counter_exc = 0;
    for cell in tqdm(cells.values()):
        traces_pos = cell['traces_pos']
        traces_radius = cell['traces_radius']
        soma_pos = cell['soma_pos']
        x_soma = convert_x(soma_pos[0])
        y_soma = convert_y(soma_pos[1])
        z_soma = convert_z(soma_pos[2])

        if traces_radius.shape==():
            counter_exc = counter_exc + 1
            continue

        radius_soma = traces_radius[0]
        soma_cube_x = convert_radius_x(radius_soma)
        soma_cube_y = convert_radius_y(radius_soma)
        soma_cube_z = convert_radius_z(radius_soma)
        radius_size_x = int(convert_radius_x(radius_soma))
        radius_size_y = int(convert_radius_y(radius_soma))
        radius_size_z = int(convert_radius_z(radius_soma))
        for i in range(-soma_cube_x, soma_cube_x+1):
            for j in range(-soma_cube_y, soma_cube_y+1):
                for k in range(-soma_cube_z, soma_cube_z+1):
                    x = x_soma+i; y = y_soma+j; z = z_soma+k
                    if x >= SizeX:
                        x = SizeX-1
                    if y >= SizeY:
                        y = SizeY-1
                    if z >= SizeZ:
                        z = SizeZ-1
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if z < 0:
                        z = 0
                    if((i/radius_size_x)**2 + (j/radius_size_y)**2 + (k/radius_size_z)**2) <= 1:
                            mask_soma[x, y, z] = 1

        for pos_no, trace_row in enumerate(traces_pos[1:, :]):
            x_trace = convert_x(trace_row[0])
            y_trace = convert_y(trace_row[1])
            z_trace = convert_z(trace_row[2])
            trace_size_x = int(convert_radius_x(traces_radius[pos_no]))
            trace_size_y = int(convert_radius_y(traces_radius[pos_no]))
            trace_size_z = int(convert_radius_z(traces_radius[pos_no]))
            for i in range(-trace_size_x, trace_size_x+1):
                for j in range(-trace_size_y, trace_size_y+1):
                    for k in range(-trace_size_z, trace_size_z+1):
                        x = x_trace+i; y = y_trace+j; z = z_trace+k
                        if x >= SizeX:
                            x = SizeX-1
                        if y >= SizeY:
                            y = SizeY-1
                        if z >= SizeZ:
                            z = SizeZ-1
                        if x < 0:
                            x = 0
                        if y < 0:
                            y = 0
                        if z < 0:
                            z = 0
                        if((i/trace_size_x)**2 + (j/trace_size_y)**2 + (k/trace_size_z)**2) <= 1:
                            mask_traces[x, y, z] = 1

    save_path_dataset_img = os.path.join(save_path_dataset, dataset_name, 'img')
    save_path_dataset_soma = os.path.join(save_path_dataset, dataset_name, 'soma')
    save_path_dataset_traces = os.path.join(save_path_dataset, dataset_name, 'traces')
    save_path_dataset_somas_pos = os.path.join(save_path_dataset, dataset_name, 'somas_pos')
    save_path_dataset_traces_pos = os.path.join(save_path_dataset, dataset_name, 'traces_pos')

    if not os.path.exists(save_path_dataset_img):
        os.makedirs(save_path_dataset_img)
    if not os.path.exists(save_path_dataset_soma):
        os.makedirs(save_path_dataset_soma)
    if not os.path.exists(save_path_dataset_traces):
        os.makedirs(save_path_dataset_traces)
    if not os.path.exists(save_path_dataset_somas_pos):
        os.makedirs(save_path_dataset_somas_pos)
    if not os.path.exists(save_path_dataset_traces_pos):
        os.makedirs(save_path_dataset_traces_pos)

    # slice img and mask_soma into window_size_x, window_size_y, window_size_z
    counter = 0
    print('- Slicing the image and masks into windows ...')
    for i_counter in tqdm(range(0, int(img.shape[0]/(window_size_x-overlap_x))+1)):
        for j_counter in range(0, int(img.shape[1]/(window_size_y-overlap_y))+1):
            for k_counter in range(0, int(img.shape[2]//(window_size_z-overlap_z))+1):
                start_x = i_counter*(window_size_x-overlap_x)
                start_y = j_counter*(window_size_y-overlap_y)
                start_z = k_counter*(window_size_z-overlap_z)
                end_x = start_x+window_size_x
                end_y = start_y+window_size_y
                end_z = start_z+window_size_z

                if start_x<0:
                    start_x = 0
                    end_x = window_size_x
                    if end_x>img.shape[0]:
                        end_x = SizeX

                if start_y<0:
                    start_y = 0
                    end_y = window_size_y
                    if end_y>img.shape[1]:
                        end_y = SizeY

                if start_z<0:
                    start_z = 0
                    end_z = window_size_z
                    if end_z>img.shape[2]:
                        end_z = SizeZ

                if end_x > img.shape[0]:
                    end_x = img.shape[0]
                    start_x = end_x - window_size_x
                    if start_x<0:
                        start_x = 0

                if end_y > img.shape[1]:
                    end_y = img.shape[1]
                    start_y = end_y - window_size_y
                    if start_y<0:
                        start_y = 0

                if end_z > img.shape[2]:
                    end_z = img.shape[2]
                    start_z = end_z - window_size_z
                    if start_z<0:
                        start_z = 0

                window_img = img[start_x:end_x, start_y:end_y, start_z:end_z]
                window_mask_soma = mask_soma[start_x:end_x, start_y:end_y, start_z:end_z]
                window_mask_traces = mask_traces[start_x:end_x, start_y:end_y, start_z:end_z]

                # save the window_img and window_mask_soma as .npz files in save_path_dataset 
                window_img = np.array(window_img, dtype=np.float32)
                window_mask_soma = np.array(window_mask_soma, dtype=bool)
                window_mask_traces = np.array(window_mask_traces, dtype=bool)

                # Histogram Equalization
                # print(f"Min and Max of window_img: {np.min(window_img)} - {np.max(window_img)}")
                # print(f"Min and Max of img: {np.min(img)} - {np.max(img)}")
                
                window_img = exposure.equalize_adapthist(window_img, kernel_size = 5) 
                
                # print(f"img.shape {img.shape} - window_size_z {window_size_z}")
                if img.shape[2]<window_size_z:
                    # print("Correcting Z dimension")
                    num_pad_rows = window_size_z - img.shape[2]
                    # print(f"Padding: {num_pad_rows} {window_size_z} {img.shape[2]}")
                    num_pad_rows_top = ceil(num_pad_rows/2)
                    num_pad_rows_bottom = floor(num_pad_rows/2)
                    # print(f"Padding: {num_pad_rows_top} - {num_pad_rows_bottom}")
                    window_img = np.pad(window_img, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
                    window_mask_soma = np.pad(window_mask_soma, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
                    window_mask_traces = np.pad(window_mask_traces, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
                if img.shape[1]<window_size_y:
                    # print("Correcting Y dimension")
                    num_pad_cols = window_size_y - img.shape[1]
                    # print(f"Padding: {num_pad_cols} {window_size_y} {img.shape[1]}")
                    num_pad_cols_top = ceil(num_pad_cols/2)
                    num_pad_cols_bottom = floor(num_pad_cols/2)
                    # print(f"Padding: {num_pad_cols_top} - {num_pad_cols_bottom}")
                    window_img = np.pad(window_img, ((0,0), (num_pad_cols_top, num_pad_cols_bottom), (0,0)), 'constant', constant_values=(0,0))
                    window_mask_soma = np.pad(window_mask_soma, ((0,0), (num_pad_cols_top, num_pad_cols_bottom), (0,0)), 'constant', constant_values=(0,0))
                    window_mask_traces = np.pad(window_mask_traces, ((0,0), (num_pad_cols_top, num_pad_cols_bottom), (0,0)), 'constant', constant_values=(0,0))
                if img.shape[0]<window_size_x:
                    # print("Correcting X dimension")
                    num_pad_cols = window_size_x - img.shape[0]
                    # print(f"Padding: {num_pad_cols} {window_size_x} {img.shape[0]}")
                    num_pad_cols_top = ceil(num_pad_cols/2)
                    num_pad_cols_bottom = floor(num_pad_cols/2)
                    # print(f"Padding: {num_pad_cols_top} - {num_pad_cols_bottom}")
                    window_img = np.pad(window_img, ((num_pad_cols_top, num_pad_cols_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))
                    window_mask_soma = np.pad(window_mask_soma, ((num_pad_cols_top, num_pad_cols_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))
                    window_mask_traces = np.pad(window_mask_traces, ((num_pad_cols_top, num_pad_cols_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))

                # print(f"shape window_img: {window_img.shape} - shape window_mask_soma: {window_mask_soma.shape} - shape window_mask_traces: {window_mask_traces.shape}")
                # break;

                soma_points = []
                for i in range(window_mask_soma.shape[0]):
                    for j in range(window_mask_soma.shape[1]):
                        for k in range(window_mask_soma.shape[2]):
                            if window_mask_soma[i,j,k]:
                                soma_points.append([i,j,k])
                
                trace_points = []
                for i in range(window_mask_traces.shape[0]):
                    for j in range(window_mask_traces.shape[1]):
                        for k in range(window_mask_traces.shape[2]):
                            if window_mask_traces[i,j,k]:
                                trace_points.append([i,j,k])

                np.savez(os.path.join(save_path_dataset_img, str(counter)+'.npz'), window_img)
                np.savez(os.path.join(save_path_dataset_soma, str(counter)+'.npz'), window_mask_soma)
                np.savez(os.path.join(save_path_dataset_traces, str(counter)+'.npz'), window_mask_traces)
                np.savez(os.path.join(save_path_dataset_somas_pos, str(counter)+'.npz'), np.array(soma_points))
                np.savez(os.path.join(save_path_dataset_traces_pos, str(counter)+'.npz'), np.array(trace_points))
                counter += 1

    metadata = {'wnidow_size_x': window_size_x, 'window_size_y': window_size_y, 
                'window_size_z': window_size_z,
                'img_size': img.shape, 
                'overlap_x': overlap_x, 'overlap_y': overlap_y, 'overlap_z': overlap_z, 
                'soma_cube_x': soma_cube_x, 'soma_cube_y': soma_cube_y, 'soma_cube_z': soma_cube_z,
                'trace_size_x': trace_size_x, 'trace_size_y': trace_size_y, 'trace_size_z': trace_size_z, 
                'x_zoom_factor': ratio_x, 'y_zoom_factor': ratio_y, 'z_zoom_factor': ratio_z,
                'num_cells': len(cells)-counter_exc
                }

    with open(os.path.join(save_path_dataset, dataset_name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


@cli.command('make_mask_trace') 
@click.option('--file_path', '-fp', type=str, required=True)
@click.option('--dataset_name', '-dn', type=str, required=True)
@click.option('--target_x_px2mu', '-tx', type=float, required=True, default = 2.5)
@click.option('--target_y_px2mu', '-ty', type=float, required=True, default = 2.5)
@click.option('--target_z_px2mu', '-tz', type=float, required=True, default = 0.8)
@click.option('--save_path_dataset', '-sp', type=str, required=True)
@click.option('--window_size_x', '-wsx', type=int, required=True, default=128)
@click.option('--window_size_y', '-wsy', type=int, required=True, default=128)
@click.option('--window_size_z', '-wsz', type=int, required=True, default=32)
@click.option('--trace_size_z', '-tsxy', type=int, required=True, default=0)
@click.option('--kernel_size', '-ks', type=int, required=True, default=5)
@click.option('--no_random_shifts', '-ni', type=int, required=True, default=1)
def make_mask_trace(file_path, dataset_name, target_x_px2mu, target_y_px2mu, target_z_px2mu, save_path_dataset, window_size_x, window_size_y, window_size_z, trace_size_z, kernel_size, no_random_shifts):
    dataset = mat73.loadmat(file_path)
    dataset = dataset['dataset']
    img = dataset['img']
    print(f"Original Size: {img.shape}")
    # check if masks are created by os.path.join(save_path_dataset, dataset_name, 'metadata.json')
    if os.path.exists(os.path.join(save_path_dataset, dataset_name, 'metadata.json')):
        print('Mask traces was already created.')
        return True

    cells = dataset['cells']
    ExtendMinX = dataset['metadata']['ExtendMinX']
    ExtendMinY = dataset['metadata']['ExtendMinY']
    ExtendMinZ = dataset['metadata']['ExtendMinZ']
    ExtendMaxX = dataset['metadata']['ExtendMaxX']
    ExtendMaxY = dataset['metadata']['ExtendMaxY']
    ExtendMaxZ = dataset['metadata']['ExtendMaxZ']
    SizeX = int(dataset['metadata']['SizeX'])
    SizeY = int(dataset['metadata']['SizeY'])
    SizeZ = int(dataset['metadata']['SizeZ'])

    origin_x_px2mu = SizeX/(ExtendMaxX-ExtendMinX)
    origin_y_px2mu = SizeY/(ExtendMaxY-ExtendMinY)
    origin_z_px2mu = SizeZ/(ExtendMaxZ-ExtendMinZ)
    print(f"origin_x_px2mu: {origin_x_px2mu} - origin_y_px2mu: {origin_y_px2mu} - origin_z_px2mu: {origin_z_px2mu}")

    size_x_new = int(SizeX/origin_x_px2mu*target_x_px2mu)
    size_y_new = int(SizeY/origin_y_px2mu*target_y_px2mu)
    size_z_new = int(SizeZ/origin_z_px2mu*target_z_px2mu)

    # maximum allowrd value for shifiting the window around the soma
    shift_x = ceil(window_size_x*0.9/2)
    shift_y = ceil(window_size_y*0.9/2)
    shift_z = ceil(window_size_z*0.9/2)

    ratio_x = size_x_new/SizeX
    ratio_y = size_y_new/SizeY
    ratio_z = size_z_new/SizeZ
    print(f"ratio_x: {ratio_x} - ratio_y: {ratio_y} - ratio_z: {ratio_z}")

    scaling_factor = (ratio_x, ratio_y, ratio_z)
    img = scipy.ndimage.zoom(img, scaling_factor)
    # img = skrescale(img, scaling_factor, anti_aliasing=True)
    img = img - np.min(img)
    img = img/np.max(img)

    SizeX = img.shape[0]
    SizeY = img.shape[1]
    SizeZ = img.shape[2]
    print(f"Size: {img.shape}")

    convert_x = lambda x: int((x-ExtendMinX)/(ExtendMaxX-ExtendMinX)*SizeX)
    convert_y = lambda y: int((y-ExtendMinY)/(ExtendMaxY-ExtendMinY)*SizeY)
    convert_z = lambda z: int((z-ExtendMinZ)/(ExtendMaxZ-ExtendMinZ)*SizeZ)
    convert_radius_x = lambda r: ceil(r/(ExtendMaxX-ExtendMinX)*SizeX)
    convert_radius_y = lambda r: ceil(r/(ExtendMaxY-ExtendMinY)*SizeY)
    convert_radius_z = lambda r: ceil(r/(ExtendMaxZ-ExtendMinZ)*SizeZ)

    print(f'- Making 3d images from somas and traces of the cells ...\n Image size: {img.shape} - Number of cells: {len(cells)} - SizeX: {SizeX} - SizeY: {SizeY} - SizeZ: {SizeZ}')
    if len(img.shape) > 3:
        return print('Error: Image has more than 3 dimensions')
    mask_soma = np.zeros(img.shape)
    mask_traces = np.zeros(img.shape)

    save_path_dataset_img = os.path.join(save_path_dataset, dataset_name, 'img')
    save_path_dataset_soma = os.path.join(save_path_dataset, dataset_name, 'soma')
    save_path_dataset_traces = os.path.join(save_path_dataset, dataset_name, 'traces')
    save_path_dataset_somas_pos = os.path.join(save_path_dataset, dataset_name, 'somas_pos')
    save_path_dataset_traces_pos = os.path.join(save_path_dataset, dataset_name, 'traces_pos')

    if not os.path.exists(save_path_dataset_img):
        os.makedirs(save_path_dataset_img)
    if not os.path.exists(save_path_dataset_soma):
        os.makedirs(save_path_dataset_soma)
    if not os.path.exists(save_path_dataset_traces):
        os.makedirs(save_path_dataset_traces)
    if not os.path.exists(save_path_dataset_somas_pos):
        os.makedirs(save_path_dataset_somas_pos)
    if not os.path.exists(save_path_dataset_traces_pos):
        os.makedirs(save_path_dataset_traces_pos)

    
    counter = 0
    counter_exc = 0
    for cell in tqdm(cells.values()):
        mask_soma = np.zeros(img.shape)
        mask_traces = np.zeros(img.shape)

        traces_pos = cell['traces_pos']
        traces_radius = cell['traces_radius']
        soma_pos = cell['soma_pos']
        x_soma = convert_x(soma_pos[0])
        y_soma = convert_y(soma_pos[1])
        z_soma = convert_z(soma_pos[2])

        if traces_radius.shape==():
            print(f"Excluded")
            counter_exc=counter_exc+1
            continue

        radius_soma = traces_radius[0]
        if radius_soma<5:
            radius_soma = 5
        soma_cube_x = convert_radius_x(radius_soma)
        soma_cube_y = convert_radius_y(radius_soma)
        soma_cube_z = convert_radius_z(radius_soma)
        radius_size_x = int(convert_radius_x(radius_soma))
        radius_size_y = int(convert_radius_y(radius_soma))
        radius_size_z = int(convert_radius_z(radius_soma))
        for i in range(-soma_cube_x, soma_cube_x+1):
            for j in range(-soma_cube_y, soma_cube_y+1):
                for k in range(-soma_cube_z, soma_cube_z+1):
                    x = x_soma+i; y = y_soma+j; z = z_soma+k
                    if x >= SizeX:
                        x = SizeX-1
                    if y >= SizeY:
                        y = SizeY-1
                    if z >= SizeZ:
                        z = SizeZ-1
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if z < 0:
                        z = 0
                    if((i/radius_size_x)**2 + (j/radius_size_y)**2 + (k/radius_size_z)**2) <= 1:
                            mask_soma[x, y, z] = 1

        for pos_no, trace_row in enumerate(traces_pos[1:, :]):
            x_trace = convert_x(trace_row[0])
            y_trace = convert_y(trace_row[1])
            z_trace = convert_z(trace_row[2])
            trace_size_x = int(convert_radius_x(traces_radius[pos_no]))
            trace_size_y = int(convert_radius_y(traces_radius[pos_no]))
            trace_size_z = int(convert_radius_z(traces_radius[pos_no]))
            for i in range(-trace_size_x, trace_size_x+1):
                for j in range(-trace_size_y, trace_size_y+1):
                    for k in range(-trace_size_z, trace_size_z+1):
                        x = x_trace+i; y = y_trace+j; z = z_trace+k
                        if x >= SizeX:
                            x = SizeX-1
                        if y >= SizeY:
                            y = SizeY-1
                        if z >= SizeZ:
                            z = SizeZ-1
                        if x < 0:
                            x = 0
                        if y < 0:
                            y = 0
                        if z < 0:
                            z = 0
                        if((i/trace_size_x)**2 + (j/trace_size_y)**2 + (k/trace_size_z)**2) <= 1:
                            mask_traces[x, y, z] = 1

        no_repeat = 0
        while no_repeat < no_random_shifts:
            if no_repeat == 0:
                random_shift_x = 0
                random_shift_y = 0
                random_shift_z = 0
            else:
                random_shift_x = np.random.randint(-shift_x, shift_x)
                random_shift_y = np.random.randint(-shift_y, shift_y)
                random_shift_z = np.random.randint(-shift_z, shift_z)
        
            start_x = floor(x_soma-(window_size_x/2)) + random_shift_x
            start_y = floor(y_soma-(window_size_y/2)) + random_shift_y
            start_z = floor(z_soma-(window_size_z/2)) + random_shift_z
            end_x = floor(x_soma+(window_size_x/2)) + random_shift_x
            end_y = floor(y_soma+(window_size_y/2)) + random_shift_y
            end_z = floor(z_soma+(window_size_z/2)) + random_shift_z

            if start_x<0:
                start_x = 0
                end_x = window_size_x
                if end_x>SizeX:
                    end_x = SizeX

            if start_y<0:
                start_y = 0
                end_y = window_size_y
                if end_y>SizeY:
                    end_y = SizeY

            if start_z<0:
                start_z = 0
                end_z = window_size_z
                if end_z>SizeZ:
                    end_z = SizeZ

            if end_x > SizeX:
                end_x = SizeX
                start_x = end_x - window_size_x
                if start_x<0:
                    start_x = 0

            if end_y > SizeY:
                end_y = SizeY
                start_y = end_y - window_size_y
                if start_y<0:
                    start_y = 0

            if end_z > SizeZ:
                end_z = SizeZ
                start_z = end_z - window_size_z
                if start_z<0:
                    start_z = 0

            window_img = deepcopy(img[start_x:end_x, start_y:end_y, start_z:end_z])
            window_mask_soma = deepcopy(mask_soma[start_x:end_x, start_y:end_y, start_z:end_z])
            window_mask_traces = deepcopy(mask_traces[start_x:end_x, start_y:end_y, start_z:end_z])

            # check if the window contains soma and image
            if len(np.unique(window_mask_soma))==1 or len(np.unique(window_img))==1:
                continue

            # adaptive histogram equalization
            window_img = exposure.equalize_adapthist(np.squeeze(window_img), kernel_size)
            # print(f"Window Size: {window_img.shape}")

            # Zero padding for Z dimension
            if img.shape[2]<window_size_z:
                # print("Correcting Z dimension")
                num_pad_rows = window_size_z - img.shape[2]
                # print(f"Padding: {num_pad_rows} {window_size_z} {img.shape[2]}")
                num_pad_rows_top = ceil(num_pad_rows/2)
                num_pad_rows_bottom = floor(num_pad_rows/2)
                # print(f"Padding: {num_pad_rows_top} - {num_pad_rows_bottom}")
                window_img = np.pad(window_img, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
                window_mask_soma = np.pad(window_mask_soma, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
                window_mask_traces = np.pad(window_mask_traces, ((0,0), (0,0), (num_pad_rows_top, num_pad_rows_bottom)), 'constant', constant_values=(0,0))
            if img.shape[1]<window_size_y:
                # print("Correcting Y dimension")
                num_pad_rows = window_size_y - img.shape[1]
                # print(f"Padding: {num_pad_rows} {window_size_y} {img.shape[1]}")
                num_pad_rows_top = ceil(num_pad_rows/2)
                num_pad_rows_bottom = floor(num_pad_rows/2)
                # print(f"Padding: {num_pad_rows_top} - {num_pad_rows_bottom}")
                window_img = np.pad(window_img, ((0,0), (num_pad_rows_top, num_pad_rows_bottom), (0,0)), 'constant', constant_values=(0,0))
                window_mask_soma = np.pad(window_mask_soma, ((0,0), (num_pad_rows_top, num_pad_rows_bottom), (0,0)), 'constant', constant_values=(0,0))
                window_mask_traces = np.pad(window_mask_traces, ((0,0), (num_pad_rows_top, num_pad_rows_bottom), (0,0)), 'constant', constant_values=(0,0))
            if img.shape[0]<window_size_x:
                # print("Correcting X dimension")
                num_pad_rows = window_size_x - img.shape[0]
                # print(f"Padding: {num_pad_rows} {window_size_x} {img.shape[0]}")
                num_pad_rows_top = ceil(num_pad_rows/2)
                num_pad_rows_bottom = floor(num_pad_rows/2)
                # print(f"Padding: {num_pad_rows_top} - {num_pad_rows_bottom}")
                window_img = np.pad(window_img, ((num_pad_rows_top, num_pad_rows_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))
                window_mask_soma = np.pad(window_mask_soma, ((num_pad_rows_top, num_pad_rows_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))
                window_mask_traces = np.pad(window_mask_traces, ((num_pad_rows_top, num_pad_rows_bottom), (0,0), (0,0)), 'constant', constant_values=(0,0))

            soma_points = []
            for i in range(window_mask_soma.shape[0]):
                for j in range(window_mask_soma.shape[1]):
                    for k in range(window_mask_soma.shape[2]):
                        if window_mask_soma[i,j,k]:
                            soma_points.append([i,j,k])

            # Extract all trace points in a list: [[x,y,z],...]
            trace_points = []
            for i in range(window_mask_soma.shape[0]):
                for j in range(window_mask_soma.shape[1]):
                    for k in range(window_mask_soma.shape[2]):
                        if window_mask_traces[i,j,k]:
                            trace_points.append([i,j,k])

            # save the window_img and window_mask_soma as .npz files in save_path_dataset 
            window_img = np.array(window_img, dtype=np.float32)
            window_mask_soma = np.array(window_mask_soma, dtype=bool)
            window_mask_traces = np.array(window_mask_traces, dtype=bool)
            soma_points = np.array(soma_points, dtype=np.float32)
            trace_points = np.array(trace_points, dtype=np.float32)

            np.savez(os.path.join(save_path_dataset_img, f'no{counter}_i{no_repeat}.npz'), window_img)
            np.savez(os.path.join(save_path_dataset_soma, f'no{counter}_i{no_repeat}.npz'), window_mask_soma)
            np.savez(os.path.join(save_path_dataset_traces, f'no{counter}_i{no_repeat}.npz'), window_mask_traces)
            np.savez(os.path.join(save_path_dataset_somas_pos, f'no{counter}_i{no_repeat}.npz'), soma_points)
            np.savez(os.path.join(save_path_dataset_traces_pos, f'no{counter}_i{no_repeat}.npz'), trace_points)

            no_repeat += 1
        counter += 1

    metadata = {'window_size_x': window_size_x, 'window_size_y': window_size_y, 
                'window_size_z': window_size_z,
                'img_size': img.shape, 
                'soma_cube_x': soma_cube_x, 'soma_cube_y': soma_cube_y, 'soma_cube_z': soma_cube_z,
                'trace_size_x': trace_size_x, 'trace_size_y': trace_size_y, 'trace_size_z': trace_size_z, 
                 'num_cells': len(cells)-counter_exc
                }
    
    with open(os.path.join(save_path_dataset, dataset_name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

@cli.command('make_mask_test_dataset')
@click.option('--file_path', '-fp', type=str, required=True)
@click.option('--dataset_name', '-dn', type=str, required=True)
@click.option('--optical_zoom', '-oz', type=int, required=True)
@click.option('--save_path_dataset', '-sp', type=str, required=True)
@click.option('--window_size_x', '-wsx', type=int, required=True, default=128)
@click.option('--window_size_y', '-wsy', type=int, required=True, default=128)
@click.option('--window_size_z', '-wsz', type=int, required=True, default=16)
@click.option('--overlap_x', '-ox', type=int, required=True, default=32)
@click.option('--overlap_y', '-oy', type=int, required=True, default=32)
@click.option('--overlap_z', '-oz', type=int, required=True, default=4)
@click.option('--trace_size_z', '-tsxy', type=int, required=True, default=0)
def make_mask_test_dataset(file_path, dataset_name, optical_zoom, save_path_dataset, window_size_x, window_size_y, window_size_z, overlap_x, overlap_y, overlap_z, trace_size_z):
    dataset = mat73.loadmat(file_path)
    dataset = dataset['dataset']
    img = dataset['img']

    SizeX = int(dataset['metadata']['SizeX'])
    SizeY = int(dataset['metadata']['SizeY'])
    SizeZ = int(dataset['metadata']['SizeZ'])

    print(f'- Making 3d images from somas and traces of the cells ...\n Image size: {img.shape} - SizeX: {SizeX} - SizeY: {SizeY} - SizeZ: {SizeZ}')
    if len(img.shape) > 3:
        return print('Error: Image has more than 3 dimensions')
    
    
    save_path_dataset_img = os.path.join(save_path_dataset, dataset_name, 'img')

    if not os.path.exists(save_path_dataset_img):
        os.makedirs(save_path_dataset_img)

    # slice img and mask_soma into window_size_x, window_size_y, window_size_z
    counter = 0
    print('- Slicing the image and masks into windows ...')
    for i_counter in tqdm(range(0, int(img.shape[0]/(window_size_x-overlap_x))+1)):
        for j_counter in range(0, int(img.shape[1]/(window_size_y-overlap_y))+1):
            for k_counter in range(0, int(img.shape[2]//(window_size_z-overlap_z))+1):
                start_x = i_counter*(window_size_x-overlap_x)
                start_y = j_counter*(window_size_y-overlap_y)
                start_z = k_counter*(window_size_z-overlap_z)
                end_x = start_x+window_size_x
                end_y = start_y+window_size_y
                end_z = start_z+window_size_z
                if end_x > img.shape[0]:
                    end_x = img.shape[0]
                    start_x = end_x - window_size_x

                if end_y > img.shape[1]:
                    end_y = img.shape[1]
                    start_y = end_y - window_size_y

                if end_z > img.shape[2]:
                    end_z = img.shape[2]
                    start_z = end_z - window_size_z

                window_img = img[start_x:end_x, start_y:end_y, start_z:end_z]

                # save the window_img and window_mask_soma as .npz files in save_path_dataset 
                window_img = np.array(window_img, dtype=np.uint8)
                np.savez(os.path.join(save_path_dataset_img, str(counter)+'.npz'), window_img)
                counter += 1

    

    metadata = {'window_size_x': window_size_x, 'window_size_y': window_size_y, 
                'window_size_z': window_size_z, 'optical_zoom': optical_zoom,
                'img_size': img.shape, 
                'overlap_x': overlap_x, 'overlap_y': overlap_y, 'overlap_z': overlap_z,
                }
    
    with open(os.path.join(save_path_dataset, dataset_name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

@cli.command('augment_datasets_in_a_folder')
@click.option('--folder_path', '-fp', type=str, required=True)
@click.option('--save_path', '-sp', type=str, required=True)
@click.option('--repeat_no', '-rn', type=int, required=True, default=1, help='Number of times to repeat the augmentation')
@click.option('--postfix', '-pf', type=str, required=True, default='augmented')
@click.option('--num_workers', '-nw', type=int, required=True, default=5)
def augment_datasets_in_a_folder(folder_path, save_path, repeat_no, postfix, num_workers):
    print(f'Augmenting datasets in {folder_path}:')
    datasets_paths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)]

    include_dict = {'image', 'soma', 'trace'}
    spatial_transforms = {
        tio.RandomAffine(scales=(0.3, 0.3, 0.2), label_keys=['soma', 'trace'], degrees=0, center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.5,
        tio.RandomAffine(scales=(0, 0, 0), label_keys=['soma', 'trace'], degrees=0, translation=(-75,75,-75,75,-3,3), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'):0.5,
        tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.05,
    }

    transforms = [tio.OneOf(spatial_transforms, p=1)]
    transform = tio.Compose(transforms)

    def augment_dataset(dataset_path):
        img_datapath = os.path.join(dataset_path, 'img')
        soma_datapath = os.path.join(dataset_path, 'soma')
        traces_datapath = os.path.join(dataset_path, 'traces')
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            if 'overlap_x' not in metadata:
                metadata['overlap_x'] = 0
                metadata['overlap_y'] = 0
                metadata['overlap_z'] = 0

            cells_dataset = datasets.cellsDataset(img_datapath, soma_datapath, traces_datapath, transform=transform, load2ram=True)
            print(f'Loading dataset: {dataset_path} ==> Loaded')
        except:
            print(f'Loading dataset: {dataset_path} ==> Failed')

        len_dataset = len(cells_dataset)
        save_path_dataset = os.path.join(save_path, dataset_path.split('/')[-1]+"_"+postfix)
        save_path_dataset_img = os.path.join(save_path_dataset, 'img')
        save_path_dataset_soma = os.path.join(save_path_dataset, 'soma')
        save_path_dataset_traces = os.path.join(save_path_dataset, 'traces')
        if not os.path.exists(save_path_dataset):
            os.makedirs(save_path_dataset_img)
            os.makedirs(save_path_dataset_soma)
            os.makedirs(save_path_dataset_traces)

        for repeat in tqdm(range(repeat_no)):
            for i in tqdm(range(len(cells_dataset))):
                if os.path.isfile(os.path.join(save_path_dataset_traces, str(len_dataset*(repeat)+i)+'.npz')):
                    continue
                while True:
                    sample = cells_dataset.__getitem__(i)
                    window_img = sample['image']
                    window_mask_soma = sample['soma']
                    window_mask_traces = sample['trace']

                    window_img = window_img.numpy()
                    window_mask_soma = window_mask_soma.numpy()
                    if len(np.unique(window_mask_soma))==1:
                        continue
                    window_mask_traces = window_mask_traces.numpy()
            
                    window_img = window_img/np.max(window_img)*255

                    window_mask_soma = window_mask_soma/np.max(window_mask_soma)
                    window_mask_traces = window_mask_traces/np.max(window_mask_traces)

                    window_img = np.array(window_img, dtype=np.uint8)
                    window_mask_soma = np.array(window_mask_soma, dtype=bool)
                    window_mask_traces = np.array(window_mask_traces, dtype=bool)
                    
                    window_img = np.squeeze(window_img)
                    window_mask_soma = np.squeeze(window_mask_soma)
                    window_mask_traces = np.squeeze(window_mask_traces)

                    np.savez(os.path.join(save_path_dataset_img, str(len_dataset*(repeat)+i)+'.npz'), window_img)
                    np.savez(os.path.join(save_path_dataset_soma, str(len_dataset*(repeat)+i)+'.npz'), window_mask_soma)
                    np.savez(os.path.join(save_path_dataset_traces, str(len_dataset*(repeat)+i)+'.npz'), window_mask_traces)
                    break

        metadata['augmentation_repeat_no'] = repeat_no
        with open(os.path.join(save_path_dataset, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        return True

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # augment_dataset(datasets_paths[0])
        results = executor.map(augment_dataset, datasets_paths)


@cli.command('check_datasets_in_a_folder')
@click.option('--folder_path', '-fp', type=str, required=True)
@click.option('--save_path', '-sp', type=str, required=True)
@click.option('--transform', '-t', type=bool, required=True, default=True)
def check_datasets_in_a_folder(folder_path, save_path, transform):
    
    datasets_paths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)]

    include_dict = {'image', 'soma', 'trace'}

    # if transform:
    #     spatial_transforms = {
    #         tio.RandomAffine(label_keys=['soma', 'trace', 'image'], degrees=30, translation=(-50,50,-50,50,-50,50), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.5,
    #         tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.1,
    #         tio.RandomNoise(std=(0, 0.1), include={'image'}): 0.1,
    #     }

    #     transforms = [tio.OneOf(spatial_transforms, p=1)]
    #     transform = tio.Compose(transforms)
    # else:
    #     transform = None

    spatial_transforms = {
        # tio.RandomAffine(label_keys=['soma', 'trace'], degrees=30, translation=(-20,20,-20,20,-20,20), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.7,
        # tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.2,
        # tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 0.8,
        tio.Resize((128, 128, 16), include={'image_noisy', 'soma', 'image', 'trace'}, image_interpolation='linear'): 1,
    }

    if transform:
        transforms = [tio.OneOf(spatial_transforms, p=1)]
        transform = tio.Compose(transforms)
    else:
        transform = None

    # datasets_paths = datasets_paths[165:]
    dataset_err = []
    for dataset_path in tqdm(datasets_paths):
        img_datapath = os.path.join(dataset_path, 'img')
        soma_datapath = os.path.join(dataset_path, 'soma')
        traces_datapath = os.path.join(dataset_path, 'traces')
        soma_pos_datapath = os.path.join(dataset_path, 'somas_pos')
        trace_pos_datapath = os.path.join(dataset_path, 'traces_pos')
        metadata_path = os.path.join(dataset_path, 'metadata.json')
            # load metadata (Not used)
        with open(metadata_path) as f:
            metadata = json.load(f)

        if 'overlap_x' not in metadata:
            metadata['overlap_x'] = 0
            metadata['overlap_y'] = 0
            metadata['overlap_z'] = 0

        cells_dataset = datasets.cellsDataset(images_path=img_datapath, somas_path=soma_datapath, traces_path=traces_datapath, somas_pos_path=soma_pos_datapath, 
                                              traces_pos_path=trace_pos_datapath, transform=transform)

        sample_img = cells_dataset.__getitem__(0)['image']
        imgs = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))
        somas = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))
        traces = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))
        for i in tqdm(range(len(cells_dataset))):
            sample = cells_dataset.__getitem__(i)
            img = sample['image']
            # check if img has nan
            if np.isnan(img.sum()):
                print(f'Nan in image: {i}')
            soma = sample['soma']
            trace = sample['trace']
            imgs[i] = img[0, :, :, :]
            somas[i] = soma[0, :, :, :]
            traces[i] = trace[0, :, :, :]
            # print shape of tensor sample['soma_pos']
            if np.max(np.array(sample['soma_pos'].shape)) != 3:
                dataset_err.append(dataset_path)
                dataset_err.append(i)
        imgs = utils.reconstruct_images_no_overlap(imgs)
        somas = utils.reconstruct_images_no_overlap(somas)
        traces = utils.reconstruct_images_no_overlap(traces)
    
        imgs = imgs/np.max(imgs)*255
        imgs = np.array(imgs, dtype=np.uint16)
        imgs = np.expand_dims(imgs, axis=0)

        somas = somas/np.max(somas)*255
        somas = np.array(somas, dtype=np.uint16)
        somas = np.expand_dims(somas, axis=0)

        traces = traces/np.max(traces)*255
        traces = np.array(traces, dtype=np.uint16)
        traces = np.expand_dims(traces, axis=0)

        image_all = np.concatenate([imgs, somas, traces], axis=0) # c x y z

        if not os.path.exists(os.path.join(save_path, dataset_path.split('/')[-1])):
            os.makedirs(os.path.join(save_path, dataset_path.split('/')[-1]))
        if transform is None:
            save_path_img = os.path.join(save_path, dataset_path.split('/')[-1], f'image_all.tif')
        else:
            save_path_img = os.path.join(save_path, dataset_path.split('/')[-1], f'image_all_transformed.tif')
        order = [3, 1, 2, 0] # z x y c
        image_all = np.transpose(image_all, order) 
        image_all = np.expand_dims(image_all, axis=0) # in shape of 1, Z, X, Y, C
        # tiff.imwrite(save_path_img, image_all,
        #                         bigtiff=True,
        #                         photometric='rgb',
        #                         # planarconfig='separate',
        #                         metadata={'axes': 'TCZXY'})
        # make z projection of img
        

        # Calculate projections
        print(image_all.shape)
        image_all = np.array(image_all, dtype=np.uint8)
        xy_projection = np.squeeze(np.max(image_all, axis=1))  # Max projection along Z-axis
        xz_projection = np.squeeze(np.max(image_all, axis=2)) # Max projection along Y-axis
        yz_projection = np.squeeze(np.max(image_all, axis=3))  # Max projection along X-axis
        print(xy_projection.shape, xz_projection.shape, yz_projection.shape)
        plt.imsave(os.path.join(save_path, dataset_path.split('/')[-1], 'xy_projection.jpg'), xy_projection)
        plt.imsave(os.path.join(save_path, dataset_path.split('/')[-1], 'xz_projection.jpg'), xz_projection)
        plt.imsave(os.path.join(save_path, dataset_path.split('/')[-1], 'yz_projection.jpg'), yz_projection)

        print(f'Saving: {save_path_img} ==> Done')
    print(dataset_err)



@cli.command('convert_datasets_to_nnUnet_format')
@click.option('--folder_path', '-fp', type=str, required=True)
@click.option('--save_path', '-sp', type=str, required=True)
def convert_datasets_to_nnUnet_format(folder_path, save_path):
    print('running')
    datasets_paths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)]

    include_dict = {'image', 'soma', 'trace'}

    # if transform:
    #     spatial_transforms = {
    #         tio.RandomAffine(label_keys=['soma', 'trace', 'image'], degrees=30, translation=(-50,50,-50,50,-50,50), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.5,
    #         tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.1,
    #         tio.RandomNoise(std=(0, 0.1), include={'image'}): 0.1,
    #     }

    #     transforms = [tio.OneOf(spatial_transforms, p=1)]
    #     transform = tio.Compose(transforms)
    # else:
    #     transform = None
    transform = False
    spatial_transforms = {
        # tio.RandomAffine(label_keys=['soma', 'trace'], degrees=30, translation=(-20,20,-20,20,-20,20), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.7,
        # tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.2,
        # tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 0.8,
        tio.Resize((128, 128, 16), include={'image_noisy', 'soma', 'image', 'trace'}, image_interpolation='linear'): 1,
    }

    if transform:
        transforms = [tio.OneOf(spatial_transforms, p=1)]
        transform = tio.Compose(transforms)
    else:
        transform = None

    # datasets_paths = datasets_paths[165:]
    dataset_err = []
    dataset_name = "Dataset001_train"

    save_path_folder = os.path.join(save_path, dataset_name)

    if not os.path.exists(os.path.join(save_path_folder, 'imagesTr')):
        os.makedirs(os.path.join(save_path_folder, 'imagesTr'))

    if not os.path.exists(os.path.join(save_path_folder, 'labelsTr')):
        os.makedirs(os.path.join(save_path_folder, 'labelsTr'))

    training_list = []

    for dataset_no, dataset_path in tqdm(enumerate(datasets_paths)):
        print(dataset_path)
        img_datapath = os.path.join(dataset_path, 'img')
        soma_datapath = os.path.join(dataset_path, 'soma')
        traces_datapath = os.path.join(dataset_path, 'traces')
        soma_pos_datapath = os.path.join(dataset_path, 'somas_pos')
        trace_pos_datapath = os.path.join(dataset_path, 'traces_pos')
        metadata_path = os.path.join(dataset_path, 'metadata.json')
            # load metadata (Not used)
        with open(metadata_path) as f:
            metadata = json.load(f)

        if 'overlap_x' not in metadata:
            metadata['overlap_x'] = 0
            metadata['overlap_y'] = 0
            metadata['overlap_z'] = 0

        cells_dataset = datasets.cellsDataset(images_path=img_datapath, somas_path=soma_datapath, traces_path=traces_datapath, somas_pos_path=soma_pos_datapath, 
                                              traces_pos_path=trace_pos_datapath, transform=transform)

        sample_img = cells_dataset.__getitem__(0)['image']
        imgs = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))
        somas = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))
        traces = np.zeros((len(cells_dataset), sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]))

        for i in tqdm(range(len(cells_dataset))):
            sample = cells_dataset.__getitem__(i)
            img = sample['image']
            # check if img has nan
            if np.isnan(img.sum()):
                print(f'Nan in image: {i}')
            soma = sample['soma']
            trace = sample['trace']
            imgs[i] = img[0, :, :, :]
            somas[i] = soma[0, :, :, :]
            traces[i] = trace[0, :, :, :]
            # print shape of tensor sample['soma_pos']
            if np.max(np.array(sample['soma_pos'].shape)) != 3:
                dataset_err.append(dataset_path)
                dataset_err.append(i)
        imgs = utils.reconstruct_images_no_overlap(imgs)
        somas = utils.reconstruct_images_no_overlap(somas)
        traces = utils.reconstruct_images_no_overlap(traces)
    
        imgs = imgs/np.max(imgs)*255
        imgs = np.array(imgs, dtype=np.uint16)
        imgs = np.expand_dims(imgs, axis=0)

        somas = somas/np.max(somas)
        somas = np.array(somas, dtype=np.uint16)
        somas = np.expand_dims(somas, axis=0)

        traces = traces/np.max(traces)
        traces = np.array(traces, dtype=np.uint16)
        traces = np.expand_dims(traces, axis=0)

        image_all = np.concatenate([imgs, somas, traces], axis=0) # c x y z

        order = [3, 1, 2, 0] # z x y c
        image_all = np.transpose(image_all, order) 
        image_all_img = np.squeeze(image_all[:, :, :, 0]) # in shape of Z, X, Y
        image_all_soma = np.squeeze(image_all[:, :, :, 1]) # in shape of Z, X, Y
        image_all_trace = np.squeeze(image_all[:, :, :, 2]) # in shape of Z, X, Y
        print(f'shape of img {image_all_img.shape}')
        print(f'shape of soma {image_all_soma.shape}')
        nii_image_all_img = sitk.GetImageFromArray(image_all_img)  # Auto-arranges (z, x, y) as (z, y, x) if needed
        nii_image_all_soma = sitk.GetImageFromArray(image_all_soma)  # Auto-arranges (z, x, y) as (z, y, x) if needed
        # nii_image_all_trace = sitk.GetImageFromArray(image_all_trace)  # Auto-arranges (z, x, y) as (z, y, x) if needed

        img_save_path = os.path.join(save_path_folder, 'imagesTr', f'sample_{dataset_no+1}_0000.nii.gz')
        soma_save_path = os.path.join(save_path_folder, 'labelsTr', f'sample_{dataset_no+1}.nii.gz')

        sitk.WriteImage(nii_image_all_img, img_save_path)
        sitk.WriteImage(nii_image_all_soma, soma_save_path)

        print(f'Saving: {save_path_folder} ==> Done')

        training_list.append({"image": f"./imagesTr/sample_{dataset_no+1}_0000.nii.gz", "label": f"./labelsTr/sample_{dataset_no+1}.nii.gz"})

    dataset_save_path = os.path.join(save_path_folder, 'dataset.json')

    dataset_json = {
        "name": dataset_name,
        "description": "3D cell microscopy dataset",
        "tensorImageSize": "3D",
        "release": "1.0",
        "modality": {
            "0": "FLUORESCENCE"
        },
        "labels": {
            "background": "0",
            "cell": "1"
        },
        "channel_names": {
        "0": "fluorescence"
        },
        "file_ending": ".nii.gz",
        "numTraining": len(training_list),
        "training": training_list
    }

    with open(dataset_save_path, 'w') as f:
        json.dump(dataset_json, f)

    print(dataset_err)


@cli.command('make_mask_from_tif')
@click.option('--file_path', '-fp', type=str, required=True)
@click.option('--dataset_name', '-dn', type=str, required=True)
@click.option('--optical_zoom', '-oz', type=int, required=True)
@click.option('--save_path_dataset', '-sp', type=str, required=True)
@click.option('--window_size_x', '-wsx', type=int, required=True, default=128)
@click.option('--window_size_y', '-wsy', type=int, required=True, default=128)
@click.option('--window_size_z', '-wsz', type=int, required=True, default=16)
@click.option('--overlap_x', '-ox', type=int, required=True, default=32)
@click.option('--overlap_y', '-oy', type=int, required=True, default=32)
@click.option('--overlap_z', '-oz', type=int, required=True, default=4)
@click.option('--kernel_size', '-ks', type=int, required=True, default=5)
def make_mask_from_tif(file_path, dataset_name, optical_zoom, save_path_dataset, window_size_x, window_size_y, window_size_z, overlap_x, overlap_y, overlap_z, kernel_size):
    # open tif file
    img = tiff.imread(file_path)
    img = np.array(img, dtype=np.float64)
    img = np.transpose(img, (1, 2, 0))
    img = (img-np.min(img))/(np.max(img)-np.min(img))*255
    img = np.array(img, dtype=np.uint8)
    # read metadata from tif file
    metadata = tiff.TiffFile(file_path).imagej_metadata
    save_path_dataset_img = os.path.join(save_path_dataset, dataset_name, 'img')

    if not os.path.exists(save_path_dataset_img):
        os.makedirs(save_path_dataset_img)

    # slice img and mask_soma into window_size_x, window_size_y, window_size_z
    counter = 0
    print('- Slicing the image and masks into windows ...')
    for i_counter in tqdm(range(0, int(img.shape[0]/(window_size_x-overlap_x))+1)):
        for j_counter in range(0, int(img.shape[1]/(window_size_y-overlap_y))+1):
            for k_counter in range(0, int(img.shape[2]//(window_size_z-overlap_z))+1):
                start_x = i_counter*(window_size_x-overlap_x)
                start_y = j_counter*(window_size_y-overlap_y)
                start_z = k_counter*(window_size_z-overlap_z)
                end_x = start_x+window_size_x
                end_y = start_y+window_size_y
                end_z = start_z+window_size_z
                if end_x > img.shape[0]:
                    end_x = img.shape[0]
                    start_x = end_x - window_size_x

                if end_y > img.shape[1]:
                    end_y = img.shape[1]
                    start_y = end_y - window_size_y

                if end_z > img.shape[2]:
                    end_z = img.shape[2]
                    start_z = end_z - window_size_z

                window_img = img[start_x:end_x, start_y:end_y, start_z:end_z]

                window_img = deepcopy(window_img)
                window_img = exposure.equalize_adapthist(np.squeeze(np.array(window_img)),kernel_size)*255
                
                # save the window_img and window_mask_soma as .npz files in save_path_dataset 
                window_img = np.array(window_img, dtype=np.uint8)
                np.savez(os.path.join(save_path_dataset_img, str(counter)+'.npz'), window_img)
                counter += 1


    metadata = {'window_size_x': window_size_x, 'window_size_y': window_size_y, 
                'window_size_z': window_size_z, 'optical_zoom': optical_zoom,
                'img_size': img.shape, 'overlap_x': overlap_x, 'overlap_y': overlap_y,
                'overlap_z': overlap_z
                }
    
    with open(os.path.join(save_path_dataset, dataset_name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print(np.shape(segmented_images))
    segmented_images = segmented_images[0]
    segmented_images = np.transpose(segmented_images, (2, 0, 1))
    segmented_images = segmented_images/np.max(segmented_images)*255
    segmented_images = np.array(segmented_images, dtype=np.uint16)

    tiff.imwrite('b.tif', segmented_images)

@cli.command('make_skeleton_from_tif')
@click.option('--file_path', '-fp', type=str, required=True)
@click.option('--save_path', '-sp', type=str, required=True)
def skeletonization_tif(file_path, save_path):
    img = tiff.imread(file_path)
    img = np.array(img, dtype=np.uint8)
    segments = img[:, 1, :, :, :] > 0
    segments = np.squeeze(segments) # z x y
    import kimimaro
    
    skels = kimimaro.skeletonize(
    segments, 
    teasar_params={
        "scale": 2, 
        "const": 50, # physical units
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500, # physical units
        "soma_detection_threshold": 750, # physical units
        "soma_invalidation_const": 300, # physical units
        "soma_invalidation_scale": 2,
        "max_paths": 300, # default None
    },
    # object_ids=[ ... ], # process only the specified labels
    # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
    # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
    dust_threshold=500, # skip connected components with fewer than this many voxels
    anisotropy=(32,16,16), # default True
    fix_branching=True, # default True
    fix_borders=True, # default True
    fill_holes=False, # default False
    fix_avocados=False, # default False
    progress=True, # default False, show progress bar
    parallel=2, # <= 0 all cpu, 1 single process, 2+ multiprocess
    parallel_chunk_size=100, # how many skeletons to process before updating progress bar
    )
    save_path = os.path.join(save_path, 'skeleton.swc')
    skel_str = skels[1].to_swc(save_path)

    with open(save_path, 'w') as f:
        f.write(skel_str)


@cli.command('extract_feature_maps')
@click.option('--model_path', '-mp', type=str, required=True)
@click.option('--img_path', '-ip', type=str, required=True)
@click.option('--save_path', '-sp', type=str, required=True)
def extract_feature_maps(model_path, img_path, save_path):
    model = models.CellBranchSegmentationModel()
    model.load_state_dict(torch.load(f'{model_path}'))
    model.eval()
    img = np.load(f'{img_path}')


@cli.command('visualize_model')
@click.option('--save_path', '-sp', type=str, required=True, default='model')
@click.option('--model', '-m', type=str, required=True, default='CellBranchSegmentationModel_AttentionResUnet_SomaGivenInLastLayer')
def visualize_model(save_path, model):
    from torchviz import make_dot
    model = getattr(models, model)()
    model.eval()
    x = torch.randn(1, 2, 512, 512, 32)
    y = model(x)
    g = make_dot(y)
    g.render(save_path, format='png')


@cli.command('soma_distribution_visualization')
@click.option('--folder_path', '-fp', type=str, required=True)
@click.option('--transform', '-t', type=bool, required=True, default=False)
def soma_distribution_visualization(folder_path, transform):
    datasets_paths = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path)]

    if transform:
        include_dict = {'image', 'soma', 'trace'}
        spatial_transforms = {
            tio.RandomAffine(scales=(0.3, 0.3, 0.2), label_keys=['soma', 'trace'], degrees=0, center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.33,
            tio.RandomAffine(scales=(0, 0, 0), label_keys=['soma', 'trace'], degrees=0, translation=(-75,75,-75,75,-3,3), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'):0.33,
            tio.RandomAffine(scales=(0, 0, 0), label_keys=['soma', 'trace'], degrees=(-60, 60, -60, 60, -60, 60), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'):0.33,
            tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.01,
        }

        transforms = [tio.OneOf(spatial_transforms, p=1)]
        transform = tio.Compose(transforms)

        transform_no = 1
    else:
        transform = None
        transform_no = 1


    counter = 0
    for indx, dataset_path in tqdm(enumerate(datasets_paths)):
            img_datapath = os.path.join(dataset_path, 'img')
            soma_datapath = os.path.join(dataset_path, 'soma')
            traces_datapath = os.path.join(dataset_path, 'traces')
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            # load metadata (Not used)
            with open(metadata_path) as f:
                metadata = json.load(f)

            if 'overlap_x' not in metadata:
                metadata['overlap_x'] = 0
                metadata['overlap_y'] = 0
                metadata['overlap_z'] = 0

            cells_dataset = datasets.cellsDataset(img_datapath, soma_datapath, traces_datapath, transform=transform)
            
            print(f'Loading dataset: {dataset_path} ==> Loaded')

            
            for i in tqdm(range(len(cells_dataset))):
                soma_img = cells_dataset.__getitem__(i)['soma']
                if indx == 0:
                    soma_avg_img = torch.zeros(soma_img.shape)
                soma_avg_img  = soma_avg_img + soma_img
                counter += 1
    for _ in range(transform_no):
        for indx, dataset_path in tqdm(enumerate(datasets_paths)):
            img_datapath = os.path.join(dataset_path, 'img')
            soma_datapath = os.path.join(dataset_path, 'soma')
            traces_datapath = os.path.join(dataset_path, 'traces')
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            # load metadata (Not used)
            with open(metadata_path) as f:
                metadata = json.load(f)

            if 'overlap_x' not in metadata:
                metadata['overlap_x'] = 0
                metadata['overlap_y'] = 0
                metadata['overlap_z'] = 0

            cells_dataset = datasets.cellsDataset(img_datapath, soma_datapath, traces_datapath, transform=transform)
            
            print(f'Loading dataset: {dataset_path} ==> Loaded')

            
            for i in tqdm(range(len(cells_dataset))):
                soma_img = cells_dataset.__getitem__(i)['soma']
                if indx == 0:
                    soma_avg_img = torch.zeros(soma_img.shape)
                soma_avg_img  = soma_avg_img + soma_img
                counter += 1

    soma_avg_img = soma_avg_img/counter

    soma_avg_img = soma_avg_img.numpy()
    soma_avg_img = np.squeeze(soma_avg_img)
    soma_avg_img = soma_avg_img/np.max(soma_avg_img)*255

    max_z_projection = np.max(soma_avg_img, axis=2)

    # plot them in a subplot
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.imshow(max_z_projection)
    ax.set_title('Maximum Z Intensity Projection')
    plt.tight_layout()
    plt.savefig('soma_dit_plot.png')
#     plt.show()


if __name__ == '__main__':
    cli()