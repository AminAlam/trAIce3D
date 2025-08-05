
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import re
from skimage import exposure
from copy import deepcopy, copy
from tqdm import tqdm
import warnings

class cellsDataset(Dataset):
    def __init__(self, images_path,
                    somas_path=None,
                    traces_path=None,
                    somas_pos_path=None,
                    traces_pos_path=None,
                    transform=None,
                    load2ram=False):

        self.images_path = images_path
        self.somas_path = somas_path
        self.traces_path = traces_path
        self.transform = transform
        self.somas_pos_path = somas_pos_path
        self.traces_pos_path = traces_pos_path
        self.load2ram = load2ram
        self.make_paths()
        if self.load2ram:
            self.load_all()

    def make_paths(self):
        
        if self.images_path is not None:
            self.images = os.listdir(self.images_path)
            self.images.sort(key=lambda f: int(re.sub('\D', '', f)))            
            self.images = [os.path.join(self.images_path, img) for img in self.images]

        if self.somas_path is not None:
            self.somas = os.listdir(self.somas_path)
            self.somas.sort(key=lambda f: int(re.sub('\D', '', f)))
            self.somas = [os.path.join(self.somas_path, soma) for soma in self.somas]

        if self.traces_path is not None:
            self.traces = os.listdir(self.traces_path)
            self.traces.sort(key=lambda f: int(re.sub('\D', '', f)))
            self.traces = [os.path.join(self.traces_path, trace) for trace in self.traces]

        if self.somas_pos_path is not None:
            self.somas_pos = os.listdir(self.somas_pos_path)
            self.somas_pos.sort(key=lambda f: int(re.sub('\D', '', f)))
            self.somas_pos = [os.path.join(self.somas_pos_path, soma_pos) for soma_pos in self.somas_pos]

        if self.traces_pos_path is not None:
            self.traces_pos = os.listdir(self.traces_pos_path)
            self.traces_pos.sort(key=lambda f: int(re.sub('\D', '', f)))
            self.traces_pos = [os.path.join(self.traces_pos_path, trace_pos) for trace_pos in self.traces_pos]

        # check of the number of images is the same as the number of somas
        if self.somas_path is not None:
            assert len(self.images) == len(self.somas), 'The number of images and somas is not the same'
        if self.traces_path is not None:
            assert len(self.images) == len(self.traces), 'The number of images and traces is not the same'
        if self.somas_pos_path is not None:
            assert len(self.images) == len(self.somas_pos), 'The number of images and somas_pos is not the same'
        if self.traces_pos_path is not None:
            assert len(self.images) == len(self.traces_pos), 'The number of images and traces_pos is not the same'

    def __len__(self):
        return len(self.images)

    def load_all(self):
        self.samples = []
        print('Loading the data into the RAM')
        for idx in tqdm(range(len(self.images))):
            sample = self.load_sample(idx)
            self.samples.append(sample) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.load2ram:
            sample = self.samples[idx]
        else:
            sample = self.load_sample(idx)

        if torch.isnan(sample['image']).any():
            print(f'There are nan values before transform in {idx} and {self.images[idx]}')

        if self.transform:
            sample = self.transform(sample)
            if 'soma_pos' in sample:
                try:
                    non_zero_indices_soma = torch.nonzero(sample['soma'])[:, 1:].double()
                    non_zero_indices_trace = torch.nonzero(sample['trace'])[:, 1:].double()
                    if len(non_zero_indices_soma) == 0:
                        non_zero_indices_soma_pos = non_zero_indices_trace[torch.randint(0, non_zero_indices_trace.size(0), (1,)), :]
                    else:
                        non_zero_indices_soma_pos = torch.mean(non_zero_indices_soma, 0, dtype=torch.float32)

                    distances = torch.norm(non_zero_indices_trace.unsqueeze(0) - non_zero_indices_soma_pos.unsqueeze(0), dim=2)  # (1, 3000)
                    closest_index = torch.argmin(distances, dim=1)  # (1,)
                    sample['soma_pos'] = non_zero_indices_trace[closest_index].double()
                except:
                    pass


        if torch.isnan(sample['image']).any():
            print(f'There are nan values after transform in {idx} and {self.images[idx]}')

        return sample
    
    def update_samples(self, samples_keep):

        if self.images_path is not None:
            self.images = [self.images[indx] for indx in samples_keep]

        if self.somas_path is not None:
            self.somas = [self.somas[indx] for indx in samples_keep]

        if self.traces_path is not None:
            self.traces = [self.traces[indx] for indx in samples_keep]

        if self.somas_pos_path is not None:
            self.somas_pos = [self.somas_pos[indx] for indx in samples_keep]

        if self.traces_pos_path is not None:
            self.traces_pos = [self.traces_pos[indx] for indx in samples_keep]
    
    def load_sample(self, idx):

        image_paths = self.images[idx]
        image = np.load(image_paths)

        image = image['arr_0']
        image = np.expand_dims(image, 0)
        image = torch.tensor(image, dtype=torch.float32)
        image = image/torch.max(image)

        sample = {'image': image}
        sample['image_noisy'] = deepcopy(sample['image'])

        if self.somas_path is not None:
            soma_paths = self.somas[idx]
            soma = np.load(soma_paths)
            soma = soma['arr_0']
            soma = np.expand_dims(soma, 0)
            soma = torch.tensor(soma, dtype=torch.float32)
            sample['soma'] = soma

        if self.traces_path is not None:
            trace_paths = self.traces[idx]
            trace = np.load(trace_paths)
            trace = trace['arr_0']
            trace = np.expand_dims(trace, 0)
            trace = torch.tensor(trace, dtype=torch.float32)
            sample['trace'] = trace

        if self.somas_pos_path is not None:
            soma_pos_pth = self.somas_pos[idx]
            soma_pos = np.load(soma_pos_pth)
            soma_pos = soma_pos['arr_0']
            soma_pos = np.expand_dims(soma_pos, 0)
            soma_pos = np.mean(soma_pos, 1)
            soma_pos = torch.tensor(soma_pos, dtype=torch.float32)
            sample['soma_pos'] = soma_pos

        if self.traces_pos_path is not None:
            trace_pos_pth = self.traces_pos[idx]
            trace_pos = np.load(trace_pos_pth)
            trace_pos = trace_pos['arr_0']
            trace_pos = np.expand_dims(trace_pos, 0)
            trace_pos = torch.tensor(trace_pos, dtype=torch.float32)
            sample['trace_pos'] = trace_pos
            
        if not self.transform:
            if self.somas_pos_path is not None and self.traces_pos_path is not None:
                distances = torch.norm(sample['trace_pos'] - sample['soma_pos'].unsqueeze(1), dim=2)  # (1, 3000)
                closest_index = torch.argmin(distances, dim=1)  # (1,)
                sample['soma_pos'] = sample['trace_pos'][:, closest_index, :]
                sample['soma_pos'] = sample['soma_pos'].squeeze(0)
                sample['trace_pos'] = torch.mean(trace_pos, 1)
        elif 'trace_pos' in sample:
            sample['trace_pos'] = torch.mean(trace_pos, 1)
            

        # print(f'{sample['soma_pos']}, {sample['soma_pos'].shape}')

        return sample
    

class justCellImages(Dataset):

    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.make_paths()

    def make_paths(self):
        self.images = os.listdir(self.images_path)
        self.images.sort(key=lambda f: int(re.sub('\D', '', f)))
        # load images and somas
        self.images = [os.path.join(self.images_path, img) for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_paths = self.images[idx]
        image = np.load(image_paths)
        sample = {'image': image['arr_0']}
        sample['image'] = np.expand_dims(sample['image'], 0)
        sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)

    def __getitm_path__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_paths = self.images[idx]
        return image_paths