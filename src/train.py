import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import floor
import tifffile as tiff
import click
from scipy.ndimage import label as sci_label
import functools
import math
import neptune
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torchio as tio
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.profiler import profile, ProfilerActivity
import torch.multiprocessing as mp


from loss_modules import soft_dice_cldice, clCE_loss


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy
)


import models
from utils import *
from datasets import *
from metrics import *
import VisTrans
import encoders

def setup(rank, world_size):
    # initialize the process group
    torch.backends.cudnn.benchmark = True
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
        
class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # Set initial_lr if not already set
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']

        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        print(f'---- Scheduler. Last epoch {self.last_epoch}')
        if self.last_epoch < self.warmup_epochs:
            return [(self.last_epoch * (base_lr - self.warmup_start_lr) / self.warmup_epochs + self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size_xy=11, window_size_z=5, size_average=True, channel=1):
        """
        3D SSIM Loss Module adapted for asymmetric microscopy data
        Args:
            window_size_xy: Size of gaussian window in X/Y plane (default=11)
            window_size_z: Size of gaussian window in Z direction (default=5)
            size_average: If True, average SSIM across batch (default=True)
            channel: Number of channels in input (default=1 for grayscale)
        """
        super(SSIM3DLoss, self).__init__()
        self.window_size_xy = window_size_xy
        self.window_size_z = window_size_z
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window_3d(window_size_xy, window_size_z, channel)

    def gaussian_3d_asymmetric(self, window_size_xy, window_size_z, sigma_xy=1.5, sigma_z=1.0):
        """Create 3D Gaussian kernel with different XY and Z dimensions"""
        coords_xy = torch.arange(window_size_xy, dtype=torch.float)
        coords_z = torch.arange(window_size_z, dtype=torch.float)
        
        gauss_xy = torch.exp(-((coords_xy - window_size_xy//2)**2) / (2*sigma_xy**2))
        gauss_z = torch.exp(-((coords_z - window_size_z//2)**2) / (2*sigma_z**2))
        
        gauss_x = gauss_xy.view(1, 1, 1, -1, 1)
        gauss_y = gauss_xy.view(1, 1, 1, 1, -1)
        gauss_z = gauss_z.view(1, 1, -1, 1, 1)
        
        kernel_3d = gauss_x * gauss_y * gauss_z
        return kernel_3d / kernel_3d.sum()

    def create_window_3d(self, window_size_xy, window_size_z, channel):
        """Create 3D window with asymmetric dimensions"""
        window = self.gaussian_3d_asymmetric(window_size_xy, window_size_z)
        window = window.expand(channel, 1, window_size_z, window_size_xy, window_size_xy)
        return window

    def forward(self, img1, img2):
        """
        Calculate SSIM loss between two 3D images with (B,C,H,W,D) format
        Args:
            img1, img2: Input 3D volumes (B,C,H,W,D)
        Returns:
            SSIM loss (1 - SSIM)
        """
        # Permute inputs to (B,C,D,H,W) for conv3d operation
        img1 = img1.permute(0, 1, 4, 2, 3)
        img2 = img2.permute(0, 1, 4, 2, 3)
        
        if not self.window.is_cuda and img1.is_cuda:
            self.window = self.window.cuda(img1.device)
        
        window = self.window
        
        # Constants for stability
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        # Correct padding format for conv3d: (D, H, W)
        padding = (self.window_size_z//2,    # Depth padding
                  self.window_size_xy//2,    # Height padding
                  self.window_size_xy//2)    # Width padding

        # Compute means
        mu1 = F.conv3d(img1, window, padding=padding, groups=self.channel)
        mu2 = F.conv3d(img2, window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=padding, groups=self.channel) - mu1_mu2

        # SSIM calculation
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Permute back to original format (B,C,H,W,D)
        ssim_map = ssim_map.permute(0, 1, 3, 4, 2)

        if self.size_average:
            ssim_loss = 1 - ssim_map.mean()
        else:
            ssim_loss = 1 - ssim_map.mean(1).mean(1).mean(1).mean(1)

        return ssim_loss
    
class LossFunction(nn.Module):
    def __init__(self, task_mode, weight=None, size_average=True, L1_weight=1e-6, L2_weight=1e-3, FP_weight=1e-9, dice_loss_weight=1, focal_loss_weight=0.2, tv_value_weight=0):
        super(LossFunction, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight, size_average)
        self.SSIMLoss = SSIM3DLoss()
        self.L1_weight = L1_weight
        self.L2_weight = L2_weight
        self.FP_weight = FP_weight
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.tv_value_weight = tv_value_weight
        self.task_mode = task_mode
        # task_mode = 'autoencoder', 'segmentation'
        assert self.task_mode in ['autoencoder', 'segmentation'], 'task_mode should be either autoencoder or segmentation'

    def WeigthedMSEloss(self, predicted, target):
        weight_map = torch.ones_like(target)
        weight_map[target >= 0.2] = 2
        weight_map[target >= 0.4] = 8
        weight_map[target >= 0.6] = 16
        squared_diff = (predicted - target) ** 2
        # loss = torch.mean(squared_diff * weight_map)
        loss = torch.mean(squared_diff)
        return loss

    def WeightedBCELoss(self, predicted, target):
        weight_map = torch.ones_like(target)
        weight_map[target >= 0.2] = 2
        weight_map[target >= 0.4] = 4
        bce_loss = -(target * torch.log(predicted + 1e-6) + (1 - target) * torch.log(1 - predicted + 1e-6))
        weighted_bce_loss = weight_map * bce_loss
        return torch.mean(weighted_bce_loss)

    def FocalLoss(self, predicted, target, alpha=0.25, gamma=3):
        BCE_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        return F_loss.mean()

    def DiceLoss(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice_coeff = (2 * intersection) / (union + 1e-5)  # Add a small epsilon to avoid division by zero
        dice_loss = 1 - dice_coeff
        return dice_loss
    
    def total_variation_loss_projected(self, img, weight=1):
        bs_img, _, x_img, y_img, z_img = img.shape
        projected_xy = torch.sum(img, dim=4)
        projected_xz = torch.sum(img, dim=3)
        projected_yz = torch.sum(img, dim=2)
        
        tv_x = torch.pow(projected_xy[:,:,1:,:]-projected_xy[:,:,:-1,:], 2).sum()
        tv_y = torch.pow(projected_xy[:,:,:,1:]-projected_xy[:,:,:,:-1], 2).sum()
        tv_xy = (tv_x+tv_y)**0.5/(bs_img*x_img*y_img)
        
        tv_x = torch.pow(projected_xz[:,:,1:,:]-projected_xz[:,:,:-1,:], 2).sum()
        tv_z = torch.pow(projected_xz[:,:,:,1:]-projected_xz[:,:,:,:-1], 2).sum()
        tv_xz = (tv_x+tv_z)**0.5/(bs_img*x_img*z_img)
        
        tv_y = torch.pow(projected_yz[:,:,1:,:]-projected_yz[:,:,:-1,:], 2).sum()
        tv_z = torch.pow(projected_yz[:,:,:,1:]-projected_yz[:,:,:,:-1], 2).sum()
        tv_yz = (tv_y+tv_z)**0.5/(bs_img*y_img*z_img)

        return weight*(tv_xy+tv_xz+tv_yz)
    
    def L1_regularization(self, weights, weight=1):
        return weight*torch.norm(weights, 1)
    
    def L2_regularization(self, weights, weight=1):
        return weight*torch.norm(weights, 2)

    def forward(self, predicted, target, weights=None):
        predicted = torch.sigmoid(predicted)
        if self.task_mode == 'autoencoder':
            mse_loss = self.WeigthedMSEloss(predicted, target)
            # bce_loss = 0.2*self.WeightedBCELoss(predicted, target)
            ssim_loss = 10*self.SSIMLoss(predicted, target)
            combined_loss = mse_loss + ssim_loss
            print(f'Weighted mse_loss: {mse_loss} SSIM loss: {ssim_loss}')
        elif self.task_mode == 'segmentation':
            # bce_loss = self.WeightedBCELoss(predicted, target)
            dice_loss = self.DiceLoss(predicted, target)
            # tv_value = self.total_variation_loss_projected(predicted)
            FocalLoss = self.FocalLoss(predicted, target)
            combined_loss =  FocalLoss + dice_loss
            print(f'Weighted FocalLoss: {FocalLoss}, Weighted dice_loss: {dice_loss}')
        if weights is not None:
            L1 = self.L1_regularization(weights, weight=self.L1_weight)
            # L2 = self.L2_regularization(weights, weight=self.L2_weight)
            combined_loss = combined_loss + L1 # + L2
            print(f'L1: {L1}')
        return combined_loss
    

class LossFunctionSomaSegmentation(nn.Module):
    def __init__(self, 
                 dice_weight=1.0,
                 focal_weight=1.0,
                 focal_gamma=1.8,
                 focal_alpha=0.85):
        super(LossFunctionSomaSegmentation, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, predicted, target):
        # Focal loss with higher alpha to focus more on positive class
        BCE_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * BCE_loss
        return focal_loss.mean()
    
    def dice_loss(self, predicted, target, smooth=1e-6):
        predicted = torch.sigmoid(predicted)
        # Preserve batch and channel dimensions while flattening spatial dimensions
        batch_size, channels = predicted.size()[:2]
        predicted = predicted.view(batch_size, channels, -1)
        target = target.view(batch_size, channels, -1)
        
        intersection = (predicted * target).sum(dim=2)
        union = predicted.sum(dim=2) + target.sum(dim=2)
        
        dice = (2. * intersection+smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, predicted, target):
        focal = self.focal_loss(predicted, target)
        bce_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none').mean()
        dice = self.dice_loss(predicted, target)
        # deice_test = self.dice_loss(target,  target)
        # print(f'focal loss {self.focal_weight * focal}')
        # print(f'dice loss {dice} and BCE {bce_loss}')
        total_loss = dice + bce_loss
        return total_loss
    


class LossFunctionBranchSegmentation(nn.Module):
    def __init__(self, 
                 dice_weight=1.0,
                 focal_weight=1.0,
                 focal_gamma=1.8,
                 focal_alpha=0.85):
        super(LossFunctionBranchSegmentation, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.soft_dice_cldice = soft_dice_cldice(alpha=0.05)
        self.clCE_loss = clCE_loss()

    def focal_loss(self, predicted, target):
        # Focal loss with higher alpha to focus more on positive class
        BCE_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * BCE_loss
        return focal_loss.mean()
    
    def dice_loss(self, predicted, target, smooth=1e-6):
        predicted = torch.sigmoid(predicted)
        # Preserve batch and channel dimensions while flattening spatial dimensions
        batch_size, channels = predicted.size()[:2]
        predicted = predicted.view(batch_size, channels, -1)
        target = target.view(batch_size, channels, -1)
        
        intersection = (predicted * target).sum(dim=2)
        union = predicted.sum(dim=2) + target.sum(dim=2)
        
        dice = (2. * intersection+smooth) / (union + smooth)
        return 1 - dice.mean()

    def compute_distance_weights(self, soma_coords, shape, distance_factor=10):
        B, _, D, H, W = shape
        
        # Create coordinate grids
        z, y, x = torch.meshgrid(
            torch.arange(D, device=soma_coords.device),
            torch.arange(H, device=soma_coords.device),
            torch.arange(W, device=soma_coords.device),
            indexing='ij'
        )
        
        # Stack coordinates into (D*H*W, 3)
        grid_coords = torch.stack([z, y, x], dim=-1).view(-1, 3)
        
        # Calculate distances for all batches at once: (B, D*H*W)
        distances = torch.sqrt(((grid_coords.unsqueeze(0) - soma_coords)**2).sum(dim=-1))
        
        # Normalize by max distance for each batch
        norm_distances = distances / distances.max(dim=1, keepdim=True)[0]
        
        # Convert to weights and reshape to (B, 1, D, H, W)
        weights = (0.1 + distance_factor * norm_distances).view(B, 1, D, H, W)
        
        return weights

    def forward(self, predicted, target, soma_coords, epoch=0, num_epochs=100):
        focal = self.focal_loss(predicted, target)
        # clCE_loss = 10*self.clCE_loss(predicted, target)
        weights = self.compute_distance_weights(soma_coords, predicted.shape)

        bce_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none')
        bce_loss = bce_loss*(1+torch.mul(weights, target))
        bce_loss = bce_loss.mean()
        # dice = self.dice_loss(predicted, target)
        predicted = torch.sigmoid(predicted)
        soft_c_dice_loss = self.soft_dice_cldice(target, predicted)
        # alpha = 0.2 + 0.6 * (epoch/num_epochs)  # Ranges from 0.3 to 0.7
        total_loss = soft_c_dice_loss + bce_loss + focal#+ dice # alpha*soft_c_dice_loss
        print(f'soft_c_dice_loss {soft_c_dice_loss}, bce_loss {bce_loss}, focal {focal}')
        return total_loss

class LossFunction_autoencoder(nn.Module):
    def __init__(self, task_mode, weight=None, size_average=True, L1_weight=1e-6, L2_weight=1e-3, FP_weight=1e-9, dice_loss_weight=1, focal_loss_weight=0.2, tv_value_weight=0):
        super(LossFunction_autoencoder, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight, size_average)
        self.SSIMLoss = SSIM3DLoss()
        self.L1_weight = L1_weight
        self.L2_weight = L2_weight
        self.FP_weight = FP_weight
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.tv_value_weight = tv_value_weight
        self.task_mode = task_mode
        # task_mode = 'autoencoder', 'segmentation'
        assert self.task_mode in ['autoencoder', 'segmentation'], 'task_mode should be either autoencoder or segmentation'

    def WeigthedMSEloss(self, predicted, target):
        # weight_map = torch.ones_like(target)
        # weight_map[target >= 0.2] = 2
        # weight_map[target >= 0.4] = 8
        # weight_map[target >= 0.6] = 16
        squared_diff = (predicted - target) ** 2
        # loss = torch.mean(squared_diff * weight_map)
        loss = torch.mean(squared_diff)
        return loss

    def WeightedBCELoss(self, predicted, target):
        weight_map = torch.ones_like(target)
        weight_map[target >= 0.2] = 2
        weight_map[target >= 0.4] = 4
        bce_loss = -(target * torch.log(predicted + 1e-6) + (1 - target) * torch.log(1 - predicted + 1e-6))
        weighted_bce_loss = weight_map * bce_loss
        return torch.mean(weighted_bce_loss)

    def FocalLoss(self, predicted, target, alpha=0.25, gamma=3):
        BCE_loss = F.binary_cross_entropy_with_logits(predicted, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        return F_loss.mean()

    def DiceLoss(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice_coeff = (2 * intersection) / (union + 1e-5)  # Add a small epsilon to avoid division by zero
        dice_loss = 1 - dice_coeff
        return dice_loss
    
    def total_variation_loss_projected(self, img, weight=1):
        bs_img, _, x_img, y_img, z_img = img.shape
        projected_xy = torch.sum(img, dim=4)
        projected_xz = torch.sum(img, dim=3)
        projected_yz = torch.sum(img, dim=2)
        
        tv_x = torch.pow(projected_xy[:,:,1:,:]-projected_xy[:,:,:-1,:], 2).sum()
        tv_y = torch.pow(projected_xy[:,:,:,1:]-projected_xy[:,:,:,:-1], 2).sum()
        tv_xy = (tv_x+tv_y)**0.5/(bs_img*x_img*y_img)
        
        tv_x = torch.pow(projected_xz[:,:,1:,:]-projected_xz[:,:,:-1,:], 2).sum()
        tv_z = torch.pow(projected_xz[:,:,:,1:]-projected_xz[:,:,:,:-1], 2).sum()
        tv_xz = (tv_x+tv_z)**0.5/(bs_img*x_img*z_img)
        
        tv_y = torch.pow(projected_yz[:,:,1:,:]-projected_yz[:,:,:-1,:], 2).sum()
        tv_z = torch.pow(projected_yz[:,:,:,1:]-projected_yz[:,:,:,:-1], 2).sum()
        tv_yz = (tv_y+tv_z)**0.5/(bs_img*y_img*z_img)

        return weight*(tv_xy+tv_xz+tv_yz)
    
    def L1_regularization(self, weights, weight=1):
        return weight*torch.norm(weights, 1)
    
    def L2_regularization(self, weights, weight=1):
        return weight*torch.norm(weights, 2)

    def forward(self, predicted, target, weights=None):
        predicted = torch.sigmoid(predicted)
        mse_loss = self.WeigthedMSEloss(predicted, target)
        # bce_loss = 0.2*self.WeightedBCELoss(predicted, target)
        ssim_loss = 10*self.SSIMLoss(predicted, target)
        combined_loss = mse_loss + ssim_loss
        print(f'Weighted mse_loss: {mse_loss} SSIM loss: {ssim_loss}')
        return combined_loss
        
def feed2network_autoendocder(inputs, model, label_mode, rank, mode='train', prompt_overwrite=None):
    desired_output = inputs['image'].to(rank)
    input2model = inputs['image_noisy'].to(rank)

    input2model = normalize_img(input2model)
    desired_output = normalize_img(desired_output)

    # print('checking the input for nan values')
    # print(input2model.shape, torch.isnan(input2model).any())
    if mode == 'train':
        output = model(input2model)
    else:
        with torch.no_grad():
            model.eval()
            output = model(input2model)

    # print('checking the output for nan values')
    # print(output.shape, torch.isnan(output).any())

    return output, desired_output, input2model, None

def feed2network_soma(inputs, model, label_mode, rank, mode='train', prompt_overwrite=None):
    input2model = inputs['image_noisy'].to(rank)
    desired_output = inputs['soma'].to(rank)

    input2model = normalize_img(input2model)
    # print('checking the input for nan values')
    # print(input2model.shape, torch.isnan(input2model).any())
    if mode == 'train':
        output = model(input2model)
    else:
        with torch.no_grad():
            output = model(input2model)

    # print('checking the output for nan values')
    # print(output.shape, torch.isnan(output).any())

    return output, desired_output, input2model, None

def feed2network_branch(inputs, model, label_mode, rank, mode='train', prompt_overwrite=None):
    input2model = inputs['image_noisy'].to(rank)
    desired_output = inputs['trace'].to(rank)
    input2model = normalize_img(input2model)
    prompts = inputs['soma_pos'].to(rank)

    if mode == 'train':
        output = model(input2model, prompts)
    else:
        with torch.no_grad():
            output = model(input2model, prompts)

    return output, desired_output, input2model, prompts

def save_examples(model, dataset, device, model_save_path, label_mode, epoch):
    print("Saving examples")
    len_dataset = floor(len(dataset))
    size_sample = np.squeeze(dataset.__getitem__(0)['image']).shape

    images = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)
    trace_images = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)
    segmented_image = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)

    for i in tqdm(range(0,len_dataset,2)):
        args_out = prepare_example_img(model, dataset, i, device, label_mode)

        images[i, :, :, :] = args_out['image']
        trace_images[i, :, :, :] = args_out['trace']
        segmented_image[i, :, :, :] = args_out['segmented_img']

        sample_img = np.expand_dims(images[i,:,:,:], axis=0)
        sample_trace = np.expand_dims(trace_images[i,:,:,:], axis=0)
        sample_segmented = np.expand_dims(segmented_image[i,:,:,:], axis=0)
        
        order = [3, 1, 2, 0] # z x y c
        image = np.transpose(sample_img, order)
        trace = np.transpose(sample_trace, order) 
        segmented = np.transpose(sample_segmented, order)
        # null_img = np.transpose(null_img, order)
        sample = np.concatenate([image, trace, segmented], axis=3)
        sample = np.transpose(sample, [3, 0, 1, 2])
        sample = np.expand_dims(sample, axis=0)

        for img, name in zip([sample], ['sample']): 
            name = f'{name}.tif'
            path = os.path.join(model_save_path, 'result', f'ecpoch_{epoch}', f"{i}", name)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            tiff.imwrite(path, img,
                        bigtiff=True,
                        photometric='rgb',
                        # planarconfig='separate',
                        metadata={'axes': 'TCZXY'})

def save_model(model, model_save_path, epoch, optimizer, scaler, label_mode, rank):
    """
    Save the model, optimizer, and other components' state dictionaries for DDP.
    """
    if rank == 0:  # Save only on rank 0
        print(f"Saving the model at epoch {epoch} on rank {rank}")

        # Create save directories
        for component in ['encoder', 'decoder', 'skip_connection_block', 'optimizer', 'scaler']:
            os.makedirs(os.path.join(model_save_path, component), exist_ok=True)
        if label_mode == 'branch':
            os.makedirs(os.path.join(model_save_path, 'prompt_encoder'), exist_ok=True)

        print("Directories created")

        # Save model components
        encoder_state_dict = model.module.encoder.state_dict()  # Access underlying model with `model.module`
        decoder_state_dict = model.module.decoder.state_dict()
        skip_connection_block_state_dict = model.module.skip_connection_block.state_dict()

        # Save model state dictionaries
        torch.save(encoder_state_dict, os.path.join(model_save_path, 'encoder', f"encoder_{epoch}.pth"))
        torch.save(decoder_state_dict, os.path.join(model_save_path, 'decoder', f"decoder_{epoch}.pth"))
        torch.save(skip_connection_block_state_dict, os.path.join(model_save_path, 'skip_connection_block', f"skip_connection_block_{epoch}.pth"))
        
        if label_mode == 'branch':
            prompt_encoder_state_dict = model.module.prompt_encoder.state_dict()
            torch.save(prompt_encoder_state_dict, os.path.join(model_save_path, 'prompt_encoder', f"prompt_encoder_{epoch}.pth"))
        
        print("Model components saved")

        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'optimizer', f"optimizer_{epoch}.pth"))
        print("Optimizer saved")

        # Save scaler state
        torch.save(scaler.state_dict(), os.path.join(model_save_path, 'scaler', f"scaler_{epoch}.pth"))
        print("Scaler saved")

        print("Model saving completed")

def save_model_fsdp(model, model_save_path, epoch, optimizer, scaler, label_mode, rank):
    """
    Save the model, optimizer, and other components' state dictionaries for FSDP.
    """
    if rank == 0:
        print(f"Saving the model at epoch {epoch} on rank {rank}")
        
        # Create save directories
        for component in ['encoder', 'decoder', 'skip_connection_block', 'optimizer', 'scaler']:
            os.makedirs(os.path.join(model_save_path, component), exist_ok=True)
        if label_mode == 'branch':
            os.makedirs(os.path.join(model_save_path, 'prompt_encoder'), exist_ok=True)
        
        print("Directories created")

    # Configure FSDP state dict settings
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # Save model components
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model_state_dict = model.state_dict()
        encoder_state_dict = model.encoder.state_dict()
        decoder_state_dict = model.decoder.state_dict()
        skip_connection_block_state_dict = model.skip_connection_block.state_dict()

    if rank == 0:
        # debug print
        for name, param in skip_connection_block_state_dict.items():
            if name == 'up_sampling_block_8times.up_sample.1.weight' or name == 'up_sampling_block_4times.up_sample.1.weight':
                print(f'gpu {rank} - param {name} - {param} - size {param.size()}')
        torch.save(encoder_state_dict, os.path.join(model_save_path, 'encoder', f"encoder_{epoch}.pth"))
        torch.save(decoder_state_dict, os.path.join(model_save_path, 'decoder', f"decoder_{epoch}.pth"))
        torch.save(skip_connection_block_state_dict, os.path.join(model_save_path, 'skip_connection_block', f"skip_connection_block_{epoch}.pth"))
        if label_mode == 'branch':
            prompt_encoder_state_dict = model.prompt_encoder.state_dict()
            torch.save(prompt_encoder_state_dict, os.path.join(model_save_path, 'prompt_encoder', f"prompt_encoder_{epoch}.pth"))
        print("Model components saved")

    # Save optimizer state
    full_osd = FSDP.optim_state_dict(model, optimizer)
    scaler_state_dict = scaler.state_dict()
    if rank == 0:
        torch.save(full_osd, os.path.join(model_save_path, 'optimizer', f"optimizer_{epoch}.pth"))
        print("Optimizer saved")
        torch.save(scaler_state_dict, os.path.join(model_save_path, 'scaler', f"scaler_{epoch}.pth"))
        print("Scaler saved")
        print("Model saving completed")

def load_model(model, checkpoint):
    """
    Load the model, optimizer, and other components' state dictionaries for FSDP.
    """
    state_dict = checkpoint
    model_dict = model.state_dict()
    mismatched_keys = []

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape != model_dict[k].shape:
                print(f"Ignoring '{k}' due to shape mismatch. "
                    f"Checkpoint shape: {v.shape}, Model shape: {model_dict[k].shape}")
                mismatched_keys.append(k)
            else:
                model_dict[k] = v
        else:
            print(f"Ignoring '{k}' as it's not in the model.")
            mismatched_keys.append(k)

    model.load_state_dict(model_dict, strict=False)
    return model

def check_datasets(train_loader, test_loader):
    print('checking the datasets')
    for batch_idx, inputs in tqdm(enumerate(train_loader)):
        print(batch_idx)

    for batch_idx, inputs in tqdm(enumerate(test_loader)):
        print(batch_idx)
    
    print('finished checking the datasets')

def train_single_epoch_soma(model, train_loader, label_mode, rank, criterion, optimizer, epoch, accumulation_steps, L1_weight, L2_weight, FP_weight, scaler, sampler= None, run=None, use_amp=True):
    model.train()
    ddp_loss = torch.zeros(6, device=rank) # loss, no of samples, F1_score, accuracy
    print(f'training rank {rank}')
    
    if sampler:
        sampler.set_epoch(epoch)
    
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    profile_batches = 1
    should_profile = False
    time_start = time.perf_counter()

    for batch_idx, inputs in tqdm(enumerate(train_loader)):

        # print(f'it took {time.perf_counter()-time_start} to load one batch')
        time_start = time.perf_counter()
        should_profile = batch_idx < profile_batches

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            if should_profile:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    output, desired_output, _, _ = feed2network_soma(inputs, model, label_mode, rank)
                print(f'GPU infor for rank {rank}: {prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=100)}')
                # print(f'it took {time.perf_counter()-time_start} to feed the model')
                time_start = time.perf_counter()
            else:
                output, desired_output, _, _ = feed2network_soma(inputs, model, label_mode, rank)
            loss = criterion(output, desired_output)
            # print(f'it took {time.perf_counter()-time_start} to calc loss')
            time_start = time.perf_counter()

        scaler.scale(loss).backward()
        # print(f'it took {time.perf_counter()-time_start} to backpropagate the loss')
        time_start = time.perf_counter()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        if run is not None:
            with torch.no_grad():
                ddp_loss[2] += calculate_F1_score(output, desired_output)
                ddp_loss[3] += calculate_accuracy(output, desired_output)
                run["train/loss_batch"].log(loss.item())

        if (batch_idx+1)%accumulation_steps == 0:

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        
            if run is not None:
               
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print('grad_norm is out of range')
                else:
                    run["train/grad_norm_batch"].log(grad_norm)

                f1_score = ddp_loss[2]/accumulation_steps
                if torch.isnan(f1_score) or torch.isinf(f1_score):
                    print('f1_score is out of range')
                else:
                    run["train/F1_score"].log(f1_score)
                ddp_loss[2] = 0

                accuracy = ddp_loss[3]/accumulation_steps
                if torch.isnan(accuracy) or torch.isinf(accuracy):
                    print('accuracy is out of range')
                else:
                    run["train/accuracy"].log(accuracy)
                ddp_loss[3] = 0

        # print(f'it took {time.perf_counter()-time_start} to calc metrics')
        time_start = time.perf_counter()
    # if np.isnan(grad_norm) or np.isinf(grad_norm):
    #     print('grad_norm is out of range')
    # else:
    #     run["train/grad_norm_batch"].log(grad_norm)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    del output, desired_output, loss

    return ddp_loss[0]/(ddp_loss[1])

def train_single_epoch_autoencoder(model, train_loader, label_mode, rank, criterion, optimizer, epoch, accumulation_steps, L1_weight, L2_weight, FP_weight, scaler, sampler= None, run=None, use_amp=True):
    model.train()
    ddp_loss = torch.zeros(6, device=rank) # loss, no of samples, F1_score, accuracy
    print(f'training rank {rank}')
    
    if sampler:
        sampler.set_epoch(epoch)
    
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    profile_batches = 1
    should_profile = False
    time_start = time.perf_counter()
    for batch_idx, inputs in tqdm(enumerate(train_loader)):

        # print(f'it took {time.perf_counter()-time_start} to load one batch')
        time_start = time.perf_counter()
        should_profile = batch_idx < profile_batches

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            if should_profile:
                output, desired_output, _, _ = feed2network_autoendocder(inputs, model, label_mode, rank)
                # print(f'it took {time.perf_counter()-time_start} to feed the model')
                time_start = time.perf_counter()
            else:
                output, desired_output, _, _ = feed2network_autoendocder(inputs, model, label_mode, rank)
            weights_model = None # get_weights_of_the_model(model)
            loss = criterion(output, desired_output, weights=weights_model)
            # print(f'it took {time.perf_counter()-time_start} to calc loss')
            time_start = time.perf_counter()

            scaler.scale(loss).backward()
        # print(f'it took {time.perf_counter()-time_start} to backpropagate the loss')
        time_start = time.perf_counter()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        if run is not None:
            with torch.no_grad():
                ddp_loss[2] += calculate_F1_score(output, desired_output)
                ddp_loss[3] += calculate_accuracy(output, desired_output)
                run["train/loss_batch"].log(loss.item())

        if (batch_idx+1)%accumulation_steps == 0:

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # print(f'it took {time.perf_counter()-time_start} to update the loss')
            time_start = time.perf_counter()
        
            if run is not None:
               
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print('grad_norm is out of range')
                else:
                    run["train/grad_norm_batch"].log(grad_norm)

                f1_score = ddp_loss[2]/accumulation_steps
                if torch.isnan(f1_score) or torch.isinf(f1_score):
                    print('f1_score is out of range')
                else:
                    run["train/F1_score"].log(f1_score)
                ddp_loss[2] = 0

                accuracy = ddp_loss[3]/accumulation_steps
                if torch.isnan(accuracy) or torch.isinf(accuracy):
                    print('accuracy is out of range')
                else:
                    run["train/accuracy"].log(accuracy)
                ddp_loss[3] = 0

        # print(f'it took {time.perf_counter()-time_start} to calc metrics')
        time_start = time.perf_counter()


    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    del output, desired_output, loss

    return ddp_loss[0]/(ddp_loss[1])

def train_single_epoch_branch(model, train_loader, label_mode, rank, criterion, optimizer, epoch, accumulation_steps, L1_weight, L2_weight, FP_weight, scaler, sampler= None, run=None, use_amp=True):
    model.train()
    ddp_loss = torch.zeros(6, device=rank) # loss, no of samples, F1_score, accuracy
    print(f'training rank {rank}')
    
    if sampler:
        sampler.set_epoch(epoch)
    
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    profile_batches = 1
    should_profile = False
    time_start = time.perf_counter()
    for batch_idx, inputs in tqdm(enumerate(train_loader)):
        time_start = time.perf_counter()
        should_profile = batch_idx < profile_batches

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            if should_profile:
                output, desired_output, _, soma_coords = feed2network_branch(inputs, model, label_mode, rank)
                print(f'it took {time.perf_counter()-time_start} to feed the model')
                time_start = time.perf_counter()
            else:
                output, desired_output, _, soma_coords = feed2network_branch(inputs, model, label_mode, rank)
            weights_model = None # get_weights_of_the_model(model)
            loss = criterion(output, desired_output, soma_coords, epoch, 500)
            time_start = time.perf_counter()

            scaler.scale(loss).backward()
        time_start = time.perf_counter()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        if run is not None:
            with torch.no_grad():
                ddp_loss[2] += calculate_F1_score(output, desired_output)
                ddp_loss[3] += calculate_accuracy(output, desired_output)
                run["train/loss_batch"].log(loss.item())

        if (batch_idx+1)%accumulation_steps == 0:

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            time_start = time.perf_counter()
        
            if run is not None:
               
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print('grad_norm is out of range')
                else:
                    run["train/grad_norm_batch"].log(grad_norm)

                f1_score = ddp_loss[2]/accumulation_steps
                if torch.isnan(f1_score) or torch.isinf(f1_score):
                    print('f1_score is out of range')
                else:
                    run["train/F1_score"].log(f1_score)
                ddp_loss[2] = 0

                accuracy = ddp_loss[3]/accumulation_steps
                if torch.isnan(accuracy) or torch.isinf(accuracy):
                    print('accuracy is out of range')
                else:
                    run["train/accuracy"].log(accuracy)
                ddp_loss[3] = 0

        time_start = time.perf_counter()


    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    del output, desired_output, loss

    return ddp_loss[0]/(ddp_loss[1])

def train_single_epoch_fsdp(model, train_loader, label_mode, rank, criterion, optimizer, epoch, accumulation_steps, L1_weight, L2_weight, FP_weight, scaler, sampler= None, run=None, use_amp=True):
    model.train()
    ddp_loss = torch.zeros(6).to(rank) # loss, no of samples, F1_score, accuracy
    print(f'training rank {rank}')


    if sampler:
        sampler.set_epoch(epoch)

    
    for batch_idx, inputs in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                output, desired_output, _, _ = feed2network(inputs, model, label_mode, rank)
            print(f'GPU infor for rank {rank}: {prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=100)}')
            weights_model = None # get_weights_of_the_model(model)
            loss = criterion(output, desired_output, weights=weights_model)
        output = output.detach()
        desired_output = desired_output.detach()
        scaler.scale(loss).backward()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        if run is not None and batch_idx%20 == 1:
            ddp_loss[2] += calculate_F1_score(output, desired_output)*20
            ddp_loss[3] += calculate_accuracy(output, desired_output)*20
            run["train/loss_batch"].log(loss.item())

        if (batch_idx+1)%accumulation_steps == 0:

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        if batch_idx%20 == 1:

            if run is not None:               
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print('grad_norm is out of range')
                else:
                    run["train/grad_norm_batch"].log(grad_norm)

                f1_score = ddp_loss[2]/accumulation_steps
                if torch.isnan(f1_score) or torch.isinf(f1_score):
                    print('f1_score is out of range')
                else:
                    run["train/F1_score"].log(f1_score)
                ddp_loss[2] = 0

                accuracy = ddp_loss[3]/accumulation_steps
                if torch.isnan(accuracy) or torch.isinf(accuracy):
                    print('accuracy is out of range')
                else:
                    run["train/accuracy"].log(accuracy)
                ddp_loss[3] = 0

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    return ddp_loss[0]/(ddp_loss[1])

def test_single_epoch_soma(model, test_loader, label_mode, rank, criterion, epoch, dataset_test, model_save_path, smaple_test_image_freq, test_result_save_freq, run=None):
    model.eval()
    ddp_loss = torch.zeros(4).to(rank) # loss, no of samples, F1_score, accuracy
    print(f'testing rank {rank}')

    for _, inputs in tqdm(enumerate(test_loader)):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            output, desired_output, _, _ = feed2network_soma(inputs, model, label_mode, rank, mode='test')
            loss = criterion(output, desired_output)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        ddp_loss[2] += calculate_F1_score(output, desired_output)
        ddp_loss[3] += calculate_accuracy(output, desired_output)

    if run is not None:
        f1_score = ddp_loss[2]/ddp_loss[1]
        if torch.isnan(f1_score) or torch.isinf(f1_score):
            print('f1_score is out of range')
        else:
            run["test/F1_score"].log(f1_score)
        accuracy = ddp_loss[3]/ddp_loss[1]
        if torch.isnan(accuracy) or torch.isinf(accuracy):
            print('accuracy is out of range')
        else:
            run["test/accuracy"].log(accuracy)
        test_loss = ddp_loss[0]/ddp_loss[1]
        if torch.isnan(test_loss) or torch.isinf(test_loss):
            print('test_loss is out of range')
        else:
            run["test/loss"].log(test_loss)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

def test_single_epoch_branch(model, test_loader, label_mode, rank, criterion, epoch, dataset_test, model_save_path, smaple_test_image_freq, test_result_save_freq, run=None):
    model.eval()
    ddp_loss = torch.zeros(4).to(rank) # loss, no of samples, F1_score, accuracy
    print(f'testing rank {rank}')

    for _, inputs in tqdm(enumerate(test_loader)):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            output, desired_output, _, soma_coords = feed2network_branch(inputs, model, label_mode, rank, mode='test')
            loss = criterion(output, desired_output, soma_coords, epoch, 500)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        ddp_loss[2] += calculate_F1_score(output, desired_output)
        ddp_loss[3] += calculate_accuracy(output, desired_output)

    if run is not None:
        f1_score = ddp_loss[2]/ddp_loss[1]
        if torch.isnan(f1_score) or torch.isinf(f1_score):
            print('f1_score is out of range')
        else:
            run["test/F1_score"].log(f1_score)
        accuracy = ddp_loss[3]/ddp_loss[1]
        if torch.isnan(accuracy) or torch.isinf(accuracy):
            print('accuracy is out of range')
        else:
            run["test/accuracy"].log(accuracy)
        test_loss = ddp_loss[0]/ddp_loss[1]
        if torch.isnan(test_loss) or torch.isinf(test_loss):
            print('test_loss is out of range')
        else:
            run["test/loss"].log(test_loss)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # if epoch%smaple_test_image_freq==0:
    #     save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
    #     # if test_result_save_freq is not None:
    #     #     if epoch%test_result_save_freq==0:
    #     #         save_examples(model, dataset_test, rank, model_save_path, label_mode, epoch)



def test_single_epoch_autoencoder(model, test_loader, label_mode, rank, criterion, epoch, dataset_test, model_save_path, smaple_test_image_freq, test_result_save_freq, run=None):
    model.eval()
    ddp_loss = torch.zeros(4).to(rank) # loss, no of samples, F1_score, accuracy
    print(f'testing rank {rank}')

    for _, inputs in tqdm(enumerate(test_loader)):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            output, desired_output, _, _ = feed2network_autoendocder(inputs, model, label_mode, rank, mode='test')
            loss = criterion(output, desired_output)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        ddp_loss[2] += calculate_F1_score(output, desired_output)
        ddp_loss[3] += calculate_accuracy(output, desired_output)
    
    if run is not None:
        f1_score = ddp_loss[2]/ddp_loss[1]
        if torch.isnan(f1_score) or torch.isinf(f1_score):
            print('f1_score is out of range')
        else:
            run["test/F1_score"].log(f1_score)
        accuracy = ddp_loss[3]/ddp_loss[1]
        if torch.isnan(accuracy) or torch.isinf(accuracy):
            print('accuracy is out of range')
        else:
            run["test/accuracy"].log(accuracy)
        test_loss = ddp_loss[0]/ddp_loss[1]
        if torch.isnan(test_loss) or torch.isinf(test_loss):
            print('test_loss is out of range')
        else:
            run["test/loss"].log(test_loss)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)


    # if epoch%smaple_test_image_freq==0:
    #     save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
    #     # if test_result_save_freq is not None:
    #     #     if epoch%test_result_save_freq==0:
    #     #         save_examples(model, dataset_test, rank, model_save_path, label_mode, epoch)

def prepare_example_img(model, dataset, sample_indx, device, label_mode, prompt_overwrite=None):
    sample = dataset.__getitem__(sample_indx)
    args_out = {}
    args_out['image'] = np.array(sample['image']*355, dtype=np.uint8)
    args_out['image_noisy']  = np.array(sample['image_noisy']*355, dtype=np.uint8)
    sample['image'] = torch.unsqueeze(sample['image'], 0)
    sample['image_noisy'] = torch.unsqueeze(sample['image_noisy'], 0)
    
    if label_mode == 'soma':
        args_out['soma'] = np.array(sample['soma']*255, dtype=np.uint8)
        sample['soma'] = torch.unsqueeze(sample['soma'], 0)
    
    if label_mode == 'branch':
        args_out['trace'] = np.array(sample['trace']*155, dtype=np.uint8)
        args_out['prompt'] = np.array(sample['soma_pos'])
        sample['trace'] = torch.unsqueeze(sample['trace'], 0)
        sample['soma_pos'] = torch.unsqueeze(sample['soma_pos'], 0)

    if label_mode  == 'soma':
        segmented_img, _, _, _ = feed2network_soma(sample, model, label_mode, device, mode='test', prompt_overwrite=prompt_overwrite)
    elif label_mode == 'img':
        segmented_img, _, _, _ = feed2network_autoendocder(sample, model, label_mode, device, mode='test', prompt_overwrite=prompt_overwrite)
    elif label_mode == 'branch':
        segmented_img, _, _, _ = feed2network_branch(sample, model, label_mode, device, mode='test', prompt_overwrite=prompt_overwrite)

    segmented_img = torch.sigmoid(segmented_img)
    segmented_img = np.squeeze(segmented_img.float().cpu().detach().numpy())
    segmented_img = np.array(segmented_img*455, dtype=np.uint8)
    args_out['segmented_img'] = segmented_img

    return args_out

def save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for sample_i in np.random.randint(0, len(dataset_test), 15):
            args_out = prepare_example_img(model, dataset_test, sample_i, rank, label_mode)
            order = [3, 1, 2, 0] # z x y c
            if label_mode == 'soma':
                soma_images = args_out['soma']
                soma = np.transpose(soma_images, order) 

            if label_mode == 'branch':
                trace_images = args_out['trace']
                trace = np.transpose(trace_images, order) 

                sample_prompt = args_out['prompt']
            sample_segmented = args_out['segmented_img']

            sample_segmented = np.expand_dims(sample_segmented, axis=0)

            segmented = np.transpose(sample_segmented, order)
            image = np.transpose(args_out['image_noisy'], order)
            
            # if np.max(segmented) != 0:
            #     segmented = segmented/np.max(segmented)
            # if np.max(image) != 0:
            #     image = image/np.max(image)

            if label_mode == 'soma':
                sample = np.concatenate([image, soma, segmented], axis=3)
            if label_mode == 'branch':
                sample = np.concatenate([image, trace, segmented], axis=3)
            if label_mode == 'img':
                sample = np.concatenate([image, segmented*0, segmented], axis=3)

            sample = np.transpose(sample, [3, 0, 1, 2]) # c z x y

            # calculate the maximum z intensity projection
            max_z_projection = np.max(sample, axis=1)
            # calculate the maximum x intensity projection
            max_x_projection = np.max(sample, axis=2)
            # calculate the maximum y intensity projection
            max_y_projection = np.max(sample, axis=3)

            max_z_projection = np.transpose(max_z_projection, [1, 2, 0])
            max_x_projection = np.transpose(max_x_projection, [1, 2, 0])
            max_y_projection = np.transpose(max_y_projection, [1, 2, 0])

            # plot them in a subplot
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.imshow(max_z_projection)

            if label_mode == 'branch':
                ax.text(sample_prompt[0, 1], sample_prompt[0, 0], 'X', color='silver', fontsize=45)

            ax.set_title('Maximum Z Intensity Projection')
            plt.tight_layout()

            if run is not None and rank == 0:
                save_folder = os.path.join(model_save_path, 'results_images')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_path = os.path.join(save_folder, f"max_intensity_projection_{sample_i}_epoch_{epoch}.png")
                plt.savefig(save_path)

                run["train/max_intensity_projection"].append(fig, name=f"max_intensity_projection_{sample_i}_epoch_{epoch}")
                plt.close(fig)
             
def load_dataset(datasets_target_path, transform, load2ram, label_mode):
    datasets_paths = [os.path.join(datasets_target_path, folder) for folder in os.listdir(datasets_target_path)]
    print(f'Loading datasets from {datasets_target_path}')
    datasets_list = []
    no_loaded = 0
    no_failed = 0
    traces_datapath = None
    somas_pos_datapath = None
    traces_pos_datapath = None
    for indx, dataset_path in enumerate(tqdm(datasets_paths)):
            samples_keep_path = os.path.join(dataset_path, 'samples_keep.npy')
        # try:
            img_datapath = os.path.join(dataset_path, 'img')
            if label_mode == 'soma':
                soma_datapath = os.path.join(dataset_path, 'soma')
            else:
                soma_datapath = None
            if label_mode == 'branch':
                traces_datapath = os.path.join(dataset_path, 'traces')
                somas_pos_datapath = os.path.join(dataset_path, 'somas_pos')
                traces_pos_datapath = os.path.join(dataset_path, 'traces_pos')
                soma_datapath = os.path.join(dataset_path, 'soma')
            else:
                traces_datapath = None
                somas_pos_datapath = None
                traces_pos_datapath = None
            cells_dataset_check = cellsDataset(img_datapath, soma_datapath, traces_datapath, somas_pos_datapath, traces_pos_datapath, transform=False, load2ram=load2ram)
            cells_dataset = cellsDataset(img_datapath, soma_datapath, traces_datapath, somas_pos_datapath, traces_pos_datapath, transform=transform, load2ram=load2ram)
            if traces_datapath is not None:
                if os.path.exists(samples_keep_path):
                    samples_keep = np.load(samples_keep_path)
                else:
                    samples_keep = check_dataset(cells_dataset_check)
                    np.save(samples_keep_path, samples_keep)
                cells_dataset.update_samples(samples_keep)
            
            datasets_list.append(cells_dataset)
            # print(f'Loading dataset {indx}: {dataset_path} ==> Loaded')
            no_loaded += 1
        # except:
            # print(f'Loading dataset {indx}: {dataset_path} ==> Failed')
            no_failed += 1

    print(f'- Loaded {no_loaded} datasets and failed to load {no_failed} datasets')
    final_dataset = torch.utils.data.ConcatDataset(datasets_list)
    return final_dataset


def check_dataset(dataset, volume_tresh=0.006):
    len_dataset = len(dataset)
    samples_indx_keep = []
    for sample_no in tqdm(range(len_dataset)):
        sample = dataset.__getitem__(sample_no)
        if 'trace' in sample:
            trace = sample['trace']
            volume_img = torch.sum(torch.ones_like(trace))
            volume_trace = torch.sum(trace)
            if volume_trace/volume_img < volume_tresh:
                continue

            trace_np = trace.squeeze().numpy()
            _, num_features = sci_label(trace_np)
            if num_features > 10:
                continue
            
            samples_indx_keep.append(sample_no)
        else:
            samples_indx_keep.append(sample_no)
    
    return samples_indx_keep

def make_dataloaders(datasets_path_train, datasets_path_test, train_test_ratio, batch_size, load2ram, transform, world_size, rank, label_mode):
    dataset_train = load_dataset(datasets_path_train, transform, load2ram, label_mode)
    if datasets_path_test is not None:
        dataset_test = load_dataset(datasets_path_test, transform, load2ram)
    else:
        # split dataset_train into train and test
        train_size = int(train_test_ratio * len(dataset_train))
        test_size = len(dataset_train) - train_size
        dataset_train, dataset_test = torch.utils.data.random_split(dataset_train, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    print(f'Makeing Distributed Samplers for world_size {world_size}')
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    # train_kwargs = {'batch_size': batch_size, 'sampler': sampler_train}
    # test_kwargs = {'batch_size': batch_size, 'sampler': sampler_test}
    # cuda_kawrgs = {'pin_memory': True, 'shuffle': False, 'num_workers':4, 'prefetch_factor':8}
    # train_kwargs.update(cuda_kawrgs)
    # test_kwargs.update(cuda_kawrgs)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, num_workers=8, prefetch_factor=2, pin_memory=True, persistent_workers=False, multiprocessing_context='spawn')
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=sampler_test, num_workers=8, prefetch_factor=2, pin_memory=True, persistent_workers=False,  multiprocessing_context='spawn')

    return train_loader, test_loader, dataset_train, dataset_test, sampler_train, sampler_test

@click.group(chain=True)
@click.option('--image_input_size', help='Size of the input image in format H_W_D', required=True, type=str)
@click.option('--encoder_patch_size', help="Size of the encoder patch in format H'_W'_D'", required=True, type=str)
@click.option('--encoder_emb_dim', help='Size of the encoder embedding', required=True, type=int)
@click.option('--encoder_depth', help='Depth of the encoder', required=True, type=int)
@click.option('--encoder_mlp_ratio', help='MLP ratio of the encoder', required=True, type=float)
@click.option('--encoder_transformer_num_heads', help='num of heads of th encoder transformer', required=True, type=int)
@click.option('--skip_connection_linear_proj_dim', help='Linear projection dimension of the skip connection', required=True, type=int, default=128)
@click.option('--residual_block_num_conv_layers', help='Number of convolutional layers in the residual block', required=True, type=int, default=2)
@click.option('--decoder_up_sampling_dim', help='Dimension of the decoder up sampling', required=True, type=int, default=2)
@click.option('--model_name', default='CellBranchSegmentationModel_AttentionResUnet', help='Model name', required=True, type=str)
@click.option('--epoch_start', default=0, help='Epoch to start training from (importan for saving the model to prevent overwriting)', type=int)
@click.option('--batch_size', help='Batch size', required=True, type=int)
@click.option('--num_epochs', help='Number of epochs', required=True, type=int)
@click.option('--learning_rate_start', default=1e-2, help='Learning rate start', required=True, type=float)
@click.option('--learning_rate_end', default=1e-4, help='Learning rate end', required=True, type=float)
@click.option('--lr_factor', default=0.8, help='Learning rate factor', required=True, type=float)
@click.option('--patience_epochs', default=4, help='Patience epochs', required=True, type=int)
@click.option('--accumulation_steps', default=20, help='Accumulation steps', required=True, type=int)
@click.option('--model_save_path', help='Path to save the model', required=True, type=str)
@click.option('--datasets_path_train', help='Path to the training dataset', required=True, type=str)
@click.option('--datasets_path_test', help='Path to the testing dataset', type=str, default=None)
@click.option('--transform_bool', default=False, help='Apply transformations to the dataset', required=True, type=bool)
@click.option('--train_test_ratio', help='Ratio of the training and testing dataset', type=float, default=0.8)
@click.option('--load2ram', default=False, help='Load the dataset to RAM', required=True, type=bool)
@click.option('--train_bool', default=True, help='Train the model', required=True, type=bool)
@click.option('--smaple_test_image_freq', default=100, help='Frequency of saving the test images (every n batch)', required=True, type=int)
@click.option('--test_result_save_freq', default=None, help='Frequency of saving the test results (every n epoch)', type=int)
@click.option('--save_model_freq', default=1, help='Frequency of saving the model (every n epoch)', type=int)
@click.option('--device', default='cuda', help='Device to use. Default is cuda, and if not available, cpu', required=True, type=str)
@click.option('--neptune_project', default='CellSegmentation/BranchSegmentation', help='Neptune project name', required=True, type=str)
@click.option('--l1_weight', default=1e-6, help='L1 weight', required=True, type=float)
@click.option('--l2_weight', default=1e-3, help='L2 weight', required=True, type=float)
@click.option('--scheulder_warmup_epochs', default=350, help='Warmup epochs for the scheduler', required=True, type=int)
@click.pass_context
def cli(ctx,
        image_input_size,
        encoder_patch_size,
        encoder_emb_dim,
        encoder_depth,
        encoder_mlp_ratio,
        encoder_transformer_num_heads,
        skip_connection_linear_proj_dim,
        residual_block_num_conv_layers,
        decoder_up_sampling_dim,
        model_save_path, 
        datasets_path_train, 
        datasets_path_test, 
        train_test_ratio,
        train_bool, 
        transform_bool, 
        learning_rate_start, 
        learning_rate_end, 
        patience_epochs, 
        lr_factor, 
        batch_size, 
        num_epochs, 
        accumulation_steps, 
        load2ram, 
        smaple_test_image_freq, 
        device, 
        neptune_project, 
        model_name,
        test_result_save_freq,
        epoch_start,
        save_model_freq,
        l1_weight,
        l2_weight,
        scheulder_warmup_epochs):
    
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    ctx = click.get_current_context()

    image_input_size = image_input_size.split('_')
    image_input_size = tuple(map(int, image_input_size))

    encoder_patch_size = encoder_patch_size.split('_')
    encoder_patch_size = tuple(map(int, encoder_patch_size))


    ctx.obj = {
        'image_input_size': image_input_size,
        'encoder_patch_size': encoder_patch_size,
        'encoder_emb_dim': encoder_emb_dim,
        'encoder_depth': encoder_depth,
        'encoder_mlp_ratio': encoder_mlp_ratio,
        'skip_connection_linear_proj_dim': skip_connection_linear_proj_dim,
        'residual_block_num_conv_layers': residual_block_num_conv_layers,
        'decoder_up_sampling_dim': decoder_up_sampling_dim,
        'encoder_transformer_num_heads': encoder_transformer_num_heads,
        'model_save_path': model_save_path,
        'datasets_path_train': datasets_path_train,
        'datasets_path_test': datasets_path_test,
        'train_test_ratio': train_test_ratio,
        'train_bool': train_bool,
        'transform_bool': transform_bool,
        'learning_rate_start': learning_rate_start,
        'learning_rate_end': learning_rate_end,
        'patience_epochs': patience_epochs,
        'lr_factor': lr_factor,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'accumulation_steps': accumulation_steps,
        'load2ram': load2ram,
        'smaple_test_image_freq': smaple_test_image_freq,
        'device': device,
        'neptune_project': neptune_project,
        'model_name': model_name,
        'test_result_save_freq': test_result_save_freq,
        'epoch_start': epoch_start,
        'save_model_freq': save_model_freq,
        'rank': rank,
        'world_size': world_size,
        'global_rank': global_rank,
        'l1_weight': l1_weight,
        'l2_weight': l2_weight,
        'scheulder_warmup_epochs': scheulder_warmup_epochs
    }

@cli.command('train_autoencoder')
@click.option('--encoder_load_path', default=None, help='Path to the encoder model', required=False, type=str)
@click.option('--decoder_load_path', default=None, help='Path to the decoder model', required=False, type=str)
@click.option('--skip_connection_block_load_path', default=None, help='Path to the skip connection block model', required=False, type=str)
@click.option('--optimizer_load_path', default=None, help='Path to the optimizer model', required=False, type=str)
@click.option('--scaler_load_path', default=None, help='Path to the scaler model', required=False, type=str)
@click.option('--freeze_encoder', default=False, help='Freeze the encoder', required=True, type=bool)
@click.option('--freeze_skip_connection_block', default=False, help='Freeze the skip connection block', required=True, type=bool)
@click.option('--freeze_decoder', default=False, help='Freeze the decoder', required=True, type=bool)
@click.option('--task_mode', default='autoencoder', help='Task mode', required=True, type=str)
@click.pass_context
def train_autoencoder(ctx, encoder_load_path, decoder_load_path, skip_connection_block_load_path, optimizer_load_path, scaler_load_path, freeze_encoder, freeze_skip_connection_block, freeze_decoder, task_mode):
    ctx = click.get_current_context()
    image_input_size = ctx.obj['image_input_size']
    encoder_patch_size = ctx.obj['encoder_patch_size']
    encoder_emb_dim = ctx.obj['encoder_emb_dim']
    encoder_depth = ctx.obj['encoder_depth']
    encoder_mlp_ratio = ctx.obj['encoder_mlp_ratio']
    encoder_transformer_num_heads= ctx.obj['encoder_transformer_num_heads']
    skip_connection_linear_proj_dim = ctx.obj['skip_connection_linear_proj_dim']
    residual_block_num_conv_layers = ctx.obj['residual_block_num_conv_layers']
    decoder_up_sampling_dim = ctx.obj['decoder_up_sampling_dim']
    model_save_path = ctx.obj['model_save_path']
    datasets_path_train = ctx.obj['datasets_path_train']
    datasets_path_test = ctx.obj['datasets_path_test']
    train_test_ratio = ctx.obj['train_test_ratio']
    train_bool = ctx.obj['train_bool']
    transform_bool = ctx.obj['transform_bool']
    learning_rate_start = ctx.obj['learning_rate_start']
    learning_rate_end = ctx.obj['learning_rate_end']
    patience_epochs = ctx.obj['patience_epochs']
    lr_factor = ctx.obj['lr_factor']
    batch_size = ctx.obj['batch_size']
    num_epochs = ctx.obj['num_epochs']
    accumulation_steps = ctx.obj['accumulation_steps']
    load2ram = ctx.obj['load2ram']
    smaple_test_image_freq = ctx.obj['smaple_test_image_freq']
    device = ctx.obj['device']
    neptune_project = ctx.obj['neptune_project']
    model_name = ctx.obj['model_name']
    test_result_save_freq = ctx.obj['test_result_save_freq']
    epoch_start = ctx.obj['epoch_start']
    save_model_freq = ctx.obj['save_model_freq']
    rank = ctx.obj['rank']
    world_size = ctx.obj['world_size']
    global_rank = ctx.obj['global_rank']
    L1_weight = ctx.obj['l1_weight']
    L2_weight = ctx.obj['l2_weight']
    scheulder_warmup_epochs = ctx.obj['scheulder_warmup_epochs']

    setup(rank, world_size)

    label_mode = 'img'

    FP_weight = 0 #1e-8

    if train_bool and global_rank == 0:

        run = neptune.init_run(
            project=neptune_project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            source_files=["train.py", "models.py", "VisTrans.py"]
        )

        run["parameters"] = {
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": learning_rate_end,
            "patience_epochs": patience_epochs,
            "lr_factor": lr_factor,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "load2ram": load2ram,
            "model_save_path": model_save_path,
            "transform_bool": transform_bool,
            "accumulation_steps": accumulation_steps,
            "train_bool": train_bool,
            "datasets_path_train": datasets_path_train,
            "datasets_path_test": datasets_path_test,
            "L1_weight": L1_weight,
            "L2_weight": L2_weight,
            "FP_weight": FP_weight,
            "smaple_test_image_freq": smaple_test_image_freq,
            "device": device,
            "label_mode": label_mode,
            "model_name": model_name,
            "encoder_load_path": encoder_load_path,
            "decoder_load_path": decoder_load_path,
            "optimizer_load_path": optimizer_load_path,
            "test_result_save_freq": test_result_save_freq,
            "freeze_encoder": freeze_encoder,
            "freeze_decoder": freeze_decoder,
            "train_test_ratio": train_test_ratio,
            "epoch_start": epoch_start,
            "save_model_freq": save_model_freq,
            "encoder_patch_size": encoder_patch_size,
            "encoder_emb_dim": encoder_emb_dim,
            "encoder_depth": encoder_depth,
            "encoder_mlp_ratio": encoder_mlp_ratio,
            "encoder_transformer_num_heads": encoder_transformer_num_heads,
            "skip_connection_block_load_path": skip_connection_block_load_path,
            "freeze_skip_connection_block": freeze_skip_connection_block,
            "residual_block_num_conv_layers": residual_block_num_conv_layers,
            "skip_connection_linear_proj_dim": skip_connection_linear_proj_dim,
            "decoder_up_sampling_dim": decoder_up_sampling_dim,
            "scheulder_warmup_epochs": scheulder_warmup_epochs
        }

    else:
        run = None

    # spatial_transforms = {
    #     # tio.RandomAffine(label_keys=['soma', 'trace'], degrees=30, translation=(-20,20,-20,20,-20,20), center='origin', include=include_dict, image_interpolation='bspline', label_interpolation='label_gaussian'): 0.7,
    #     # tio.RandomFlip(axes=['Left', 'right'], include=include_dict): 0.2,
    #     tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 0.8,
    # }

    # if transform_bool:
    #     transforms = [tio.OneOf(spatial_transforms, p=0.9)]
    #     transform = tio.Compose(transforms)
    # else:
    #     transform = None

    spatial_transforms = {
    # # Your other random transforms here
    # # tio.RandomAffine(...): 0.7,
    # # tio.RandomFlip(...): 0.2,
    tio.RandomAffine(label_keys=['soma', 'image'], degrees=20, translation=(-10,10,-10,10,-10,10), center='origin', include={'soma', 'image'}, image_interpolation='bspline', label_interpolation='label_gaussian'): 1,
    tio.RandomFlip(axes=['Left', 'right'], include={'image', 'soma'}): 1,

    # Existing transforms
    tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 1,
    tio.RandomBlur(std=(0, 0.5), include={'image', 'image_noisy'}): 0.5,  # Minimal blurring
    tio.RandomGamma(log_gamma=(-0.1, 0.1), include={'image', 'image_noisy'}): 0.5,  # Minimal gamma adjustment
    
    # Careful spatial transforms
    # tio.RandomElasticDeformation(max_displacement=(5, 5, 5), include={'image', 'soma'}): 0.3,  # Minimal elastic deformation
    }

    if transform_bool:
        transforms = [
            # Deterministic resize - always applied
            # tio.Resize((128, 128, 16), 
            #         include={'trace', 'image', 'image_noisy'}, 
            #         image_interpolation='linear'),
            # # Random transforms - applied with probability
            tio.OneOf(spatial_transforms, p=1)
        ]
        transform = tio.Compose(transforms)
    else:
        transform = None

    print(f'- Global rank {global_rank} is training')
    print('- Loading datasets')
    train_loader, test_loader, dataset_train, dataset_test, sampler_train, _ = make_dataloaders(datasets_path_train, 
                                                                                datasets_path_test, 
                                                                                train_test_ratio, 
                                                                                batch_size, 
                                                                                load2ram, 
                                                                                transform, 
                                                                                world_size=world_size, 
                                                                                rank=rank,
                                                                                label_mode=label_mode)

    # Model Definition
    print(f'img_size {image_input_size}, encoder_patch_size {encoder_patch_size}')
    encoder = VisTrans.UNet_ImageEncoderViT3D(img_size=image_input_size, 
                                            patch_size=encoder_patch_size,
                                            in_chans=1,
                                            embed_dim=encoder_emb_dim,
                                            depth=encoder_depth,
                                            num_heads=encoder_transformer_num_heads,
                                            mlp_ratio=encoder_mlp_ratio,
                                            dropout_rate=0.1,
                                            device=rank)
    
    num_patches = encoder.num_patches

    skip_connection_block = models.SkipConnectionBlock(input_img_size=image_input_size,
                                                            num_patches=num_patches,
                                                            linear_proj_dim=skip_connection_linear_proj_dim,
                                                            up_sampling_dim=decoder_up_sampling_dim)
    
    decoder = models.U_Net_decoder(skip_connections=True, 
                                        num_patches=num_patches,
                                        up_sampling_dim=decoder_up_sampling_dim,
                                        num_conv_layers=residual_block_num_conv_layers)


    
    if encoder_load_path is not None and encoder_load_path != 'None':
        # encoder.load_state_dict(torch.load(encoder_load_path, map_location=f'cuda:{rank}'))
        print(f'- Encoder loaded from {encoder_load_path}')

        encoder = load_model(encoder, torch.load(encoder_load_path, map_location='cpu', weights_only=True))

    if decoder_load_path is not None and decoder_load_path != 'None':
        # decoder.load_state_dict(torch.load(decoder_load_path, map_location=f'cuda:{rank}'))
        print(f'- Decoder loaded from {decoder_load_path}')

        decoder = load_model(decoder, torch.load(decoder_load_path, map_location='cpu', weights_only=True))

    if skip_connection_block_load_path is not None and skip_connection_block_load_path != 'None':
        # skip_connection_block.load_state_dict(torch.load(skip_connection_block_load_path, map_location=f'cuda:{rank}'))
        print(f'- Skip connection block loaded from {skip_connection_block_load_path}')

        skip_connection_block = load_model(skip_connection_block, torch.load(skip_connection_block_load_path, map_location='cpu', weights_only=True))

    model = eval(f'models.{model_name}')(encoder, decoder, skip_connection_block=skip_connection_block,
                            freeze_encoder=freeze_encoder, 
                            freeze_skip_connection_block=freeze_skip_connection_block, 
                            freeze_decoder=freeze_decoder)

    # model = eval(f'models.{model_name}')(encoder, decoder)

    # model = models.CellSomaSegmentationModel()

                            

    model_total_params = sum(p.numel() for p in model.parameters())
    print("- Total parameters:", model_total_params)
    encoder_total_params = sum(p.numel() for p in encoder.parameters())
    print("- Encoder parameters:", encoder_total_params)
    decoder_total_params = sum(p.numel() for p in decoder.parameters())
    print("- Decoder parameters:", decoder_total_params)
    skip_connection_block_total_params = sum(p.numel() for p in skip_connection_block.parameters())
    print("- Skip connection block parameters:", skip_connection_block_total_params)
    total_no_training_data = len(dataset_train)
    print("- Total number of training data:", total_no_training_data)

    ##### FSDP
    # torch.cuda.set_device(rank)
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )

    # model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device())

    ##### DDP
    model = model.to(rank)  # Move model to the appropriate GPU
    model = DDP(model, device_ids=[rank])  # Wrap with DDP


    criterion = LossFunction_autoencoder(FP_weight=FP_weight, L1_weight=L1_weight, L2_weight=L2_weight, task_mode=task_mode)
    criterion = criterion.to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_start, weight_decay=L2_weight)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if optimizer_load_path is not None and optimizer_load_path != 'None':
        ##### FSDP
        # full_osd = torch.load(optimizer_load_path, map_location='cpu', weights_only=True)
        # sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model) 
        # optimizer.load_state_dict(sharded_osd)

        ##### DDP
        optimizer_state_dict = torch.load(optimizer_load_path, map_location='cpu')  # Load the full optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        print(f'- Optimizer loaded from {optimizer_load_path}')

    if scaler_load_path is not None and scaler_load_path != 'None': 
        scaler.load_state_dict(torch.load(scaler_load_path, map_location='cpu', weights_only=True))
        print(f'- Scaler loaded from {scaler_load_path}')
    
    # scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=scheulder_warmup_epochs, max_epochs=num_epochs, warmup_start_lr=1e-4, eta_min=learning_rate_end, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=lr_factor, min_lr=learning_rate_end, verbose=True)

    if train_bool:

        # init_start_event.record() ##### FSDP
        for epoch in tqdm(range(epoch_start, num_epochs)):


            save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
            test_single_epoch_autoencoder(model=model, 
                test_loader=test_loader,
                label_mode=label_mode,
                rank=rank,
                criterion=criterion,
                epoch=epoch,
                dataset_test=dataset_test,
                model_save_path=model_save_path,
                smaple_test_image_freq=smaple_test_image_freq,
                test_result_save_freq=test_result_save_freq,
                run=run)
                # dist.barrier() ##### FSDP

            total_loss_train = train_single_epoch_autoencoder(model=model,
                                        train_loader=train_loader,
                                        label_mode=label_mode,
                                        rank=rank,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        epoch=epoch,
                                        accumulation_steps=accumulation_steps,
                                        L1_weight=L1_weight,
                                        L2_weight=L2_weight,
                                        FP_weight=FP_weight,
                                        scaler=scaler,
                                        sampler=sampler_train,
                                        run=run,
                                        use_amp=use_amp)

            scheduler.step(total_loss_train)
            if run is not None:
                run["train/lr"].log(scheduler.get_last_lr()[0])
                if torch.isnan(total_loss_train) or torch.isinf(total_loss_train):
                    print(f'Loss is NaN or Inf. Skipping epoch {epoch}')
                else:
                    run["train/loss"].log(total_loss_train)

            
            
            if epoch%save_model_freq==0:
                print(f'- Saving model at epoch {epoch}')

                save_model(model, model_save_path, epoch, optimizer, scaler, label_mode, rank)
                # dist.barrier() ##### FSDP


        # init_end_event.record() ##### FSDP
        print("Training complete.")
        if run is not None and rank == 0:
            run.stop()


@cli.command('train_soma_segmentation')
@click.option('--encoder_load_path', default=None, help='Path to the encoder model', required=False, type=str)
@click.option('--decoder_load_path', default=None, help='Path to the decoder model', required=False, type=str)
@click.option('--skip_connection_block_load_path', default=None, help='Path to the skip connection block model', required=False, type=str)
@click.option('--optimizer_load_path', default=None, help='Path to the optimizer model', required=False, type=str)
@click.option('--scaler_load_path', default=None, help='Path to the scaler model', required=False, type=str)
@click.option('--freeze_encoder', default=False, help='Freeze the encoder', required=True, type=bool)
@click.option('--freeze_skip_connection_block', default=False, help='Freeze the skip connection block', required=True, type=bool)
@click.option('--freeze_decoder', default=False, help='Freeze the decoder', required=True, type=bool)
@click.option('--task_mode', default='segmentation', help='Task mode', required=True, type=str)
@click.pass_context
def train_soma_segmentation(ctx, encoder_load_path, decoder_load_path, skip_connection_block_load_path, optimizer_load_path, scaler_load_path, freeze_encoder, freeze_skip_connection_block, freeze_decoder, task_mode):
    ctx = click.get_current_context()
    image_input_size = ctx.obj['image_input_size']
    encoder_patch_size = ctx.obj['encoder_patch_size']
    encoder_emb_dim = ctx.obj['encoder_emb_dim']
    encoder_depth = ctx.obj['encoder_depth']
    encoder_mlp_ratio = ctx.obj['encoder_mlp_ratio']
    encoder_transformer_num_heads = ctx.obj['encoder_transformer_num_heads']
    skip_connection_linear_proj_dim = ctx.obj['skip_connection_linear_proj_dim']
    residual_block_num_conv_layers = ctx.obj['residual_block_num_conv_layers']
    decoder_up_sampling_dim = ctx.obj['decoder_up_sampling_dim']
    model_save_path = ctx.obj['model_save_path']
    datasets_path_train = ctx.obj['datasets_path_train']
    datasets_path_test = ctx.obj['datasets_path_test']
    train_test_ratio = ctx.obj['train_test_ratio']
    train_bool = ctx.obj['train_bool']
    transform_bool = ctx.obj['transform_bool']
    learning_rate_start = ctx.obj['learning_rate_start']
    learning_rate_end = ctx.obj['learning_rate_end']
    patience_epochs = ctx.obj['patience_epochs']
    lr_factor = ctx.obj['lr_factor']
    batch_size = ctx.obj['batch_size']
    num_epochs = ctx.obj['num_epochs']
    accumulation_steps = ctx.obj['accumulation_steps']
    load2ram = ctx.obj['load2ram']
    smaple_test_image_freq = ctx.obj['smaple_test_image_freq']
    device = ctx.obj['device']
    neptune_project = ctx.obj['neptune_project']
    model_name = ctx.obj['model_name']
    test_result_save_freq = ctx.obj['test_result_save_freq']
    epoch_start = ctx.obj['epoch_start']
    save_model_freq = ctx.obj['save_model_freq']
    rank = ctx.obj['rank']
    world_size = ctx.obj['world_size']
    global_rank = ctx.obj['global_rank']
    L1_weight = ctx.obj['l1_weight']
    L2_weight = ctx.obj['l2_weight']
    scheulder_warmup_epochs = ctx.obj['scheulder_warmup_epochs']

    setup(rank, world_size)

    label_mode = 'soma'

    FP_weight = 0 #1e-8

    if train_bool and global_rank == 0:

        run = neptune.init_run(
            project=neptune_project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            source_files=["train.py", "models.py", "VisTrans.py"]
        )

        run["parameters"] = {
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": learning_rate_end,
            "patience_epochs": patience_epochs,
            "lr_factor": lr_factor,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "load2ram": load2ram,
            "model_save_path": model_save_path,
            "transform_bool": transform_bool,
            "accumulation_steps": accumulation_steps,
            "train_bool": train_bool,
            "datasets_path_train": datasets_path_train,
            "datasets_path_test": datasets_path_test,
            "L1_weight": L1_weight,
            "L2_weight": L2_weight,
            "FP_weight": FP_weight,
            "smaple_test_image_freq": smaple_test_image_freq,
            "device": device,
            "label_mode": label_mode,
            "model_name": model_name,
            "encoder_load_path": encoder_load_path,
            "decoder_load_path": decoder_load_path,
            "optimizer_load_path": optimizer_load_path,
            "test_result_save_freq": test_result_save_freq,
            "freeze_encoder": freeze_encoder,
            "freeze_decoder": freeze_decoder,
            "train_test_ratio": train_test_ratio,
            "epoch_start": epoch_start,
            "save_model_freq": save_model_freq,
            "encoder_patch_size": encoder_patch_size,
            "encoder_emb_dim": encoder_emb_dim,
            "encoder_depth": encoder_depth,
            "encoder_mlp_ratio": encoder_mlp_ratio,
            "encoder_transformer_num_heads": encoder_transformer_num_heads,
            "skip_connection_block_load_path": skip_connection_block_load_path,
            "freeze_skip_connection_block": freeze_skip_connection_block,
            "residual_block_num_conv_layers": residual_block_num_conv_layers,
            "skip_connection_linear_proj_dim": skip_connection_linear_proj_dim,
            "decoder_up_sampling_dim": decoder_up_sampling_dim,
            "scheulder_warmup_epochs": scheulder_warmup_epochs
        }

    else:
        run = None

    spatial_transforms = {
    # # Your other random transforms here
    # # tio.RandomAffine(...): 0.7,
    # # tio.RandomFlip(...): 0.2,
    # tio.RandomAffine(label_keys=['soma', 'image'], degrees=20, translation=(-10,10,-10,10,-10,10), center='origin', include={'soma', 'image'}, image_interpolation='bspline', label_interpolation='label_gaussian'): 1,
    # tio.RandomFlip(axes=['Left', 'right'], include={'image', 'soma'}): 1,
    # tio.RandomNoise(std=(0, 0.002), include={'image'}): 1,

    # Existing transforms
    tio.RandomAffine(label_keys=['soma'], 
                     degrees=20, 
                     translation=(-10,10,-10,10,-10,10), 
                     center='origin', 
                     include={'soma', 'image', 'image_noisy'}, 
                     image_interpolation='linear', 
                     label_interpolation='linear'): 1,
    tio.RandomFlip(axes=['Left', 'right'], include={'image', 'soma', 'image_noisy'}): 1,
    tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 1,
    # Gentle intensity transforms
    tio.RandomBlur(std=(0, 0.5), include={'image', 'image_noisy'}): 0.5,  # Minimal blurring
    tio.RandomGamma(log_gamma=(-0.1, 0.1), include={'image', 'image_noisy'}): 0.5,  # Minimal gamma adjustment
    
    # Careful spatial transforms
    tio.RandomElasticDeformation(max_displacement=(0.5, 0.5, 0.5), include={'image_noisy', 'image', 'soma'}): 0.3,  # Minimal elastic deformation
    }

    if transform_bool:
        transforms = [
            # Deterministic resize - always applied
            # tio.Resize((128, 128, 16), 
            #         include={'soma', 'image', 'image_noisy'}, 
            #         image_interpolation='linear'),
            # Random transforms - applied with probability
            tio.OneOf(spatial_transforms, p=1)
        ]
        transform = tio.Compose(transforms)
    else:
        transform = None

    print(f'- Global rank {global_rank} is training')
    print('- Loading datasets')
    train_loader, test_loader, dataset_train, dataset_test, sampler_train, _ = make_dataloaders(datasets_path_train, 
                                                                                datasets_path_test, 
                                                                                train_test_ratio, 
                                                                                batch_size, 
                                                                                load2ram, 
                                                                                transform, 
                                                                                world_size=world_size, 
                                                                                rank=rank,
                                                                                label_mode=label_mode)
    # check_datasets(train_loader, test_loader)

    # Model Definition
    encoder = VisTrans.UNet_ImageEncoderViT3D(img_size=image_input_size, 
                                            patch_size=encoder_patch_size,
                                            in_chans=1,
                                            embed_dim=encoder_emb_dim,
                                            depth=encoder_depth,
                                            num_heads=encoder_transformer_num_heads,
                                            mlp_ratio=encoder_mlp_ratio,
                                            dropout_rate=0.1,
                                            device=rank)
    
    num_patches = encoder.num_patches

    skip_connection_block = models.SkipConnectionBlock(input_img_size=image_input_size,
                                                            num_patches=num_patches,
                                                            linear_proj_dim=skip_connection_linear_proj_dim,
                                                            up_sampling_dim=decoder_up_sampling_dim)
    
    decoder = models.U_Net_decoder(skip_connections=True, 
                                        num_patches=num_patches,
                                        up_sampling_dim=decoder_up_sampling_dim,
                                        num_conv_layers=residual_block_num_conv_layers)

    
    if encoder_load_path is not None and encoder_load_path != 'None':
        encoder = load_model(encoder, torch.load(encoder_load_path, map_location='cpu', weights_only=True))

    if decoder_load_path is not None and decoder_load_path != 'None':
        decoder = load_model(decoder, torch.load(decoder_load_path, map_location='cpu', weights_only=True))

    if skip_connection_block_load_path is not None and skip_connection_block_load_path != 'None':
        skip_connection_block = load_model(skip_connection_block, torch.load(skip_connection_block_load_path, map_location='cpu', weights_only=True))


    model = eval(f'models.{model_name}')(encoder, decoder, skip_connection_block=skip_connection_block,
                            freeze_encoder=freeze_encoder, 
                            freeze_skip_connection_block=freeze_skip_connection_block, 
                            freeze_decoder=freeze_decoder)

    # model = models.CellSomaSegmentationModel()
    # model = eval(f'models.{model_name}')(encoder, decoder)


    model_total_params = sum(p.numel() for p in model.parameters())
    print("- Total parameters:", model_total_params)

    encoder_total_params = sum(p.numel() for p in encoder.parameters())
    print("- Encoder parameters:", encoder_total_params)

    decoder_total_params = sum(p.numel() for p in decoder.parameters())
    print("- Decoder parameters:", decoder_total_params)

    skip_connection_block_total_params = sum(p.numel() for p in skip_connection_block.parameters())
    print("- Skip connection block parameters:", skip_connection_block_total_params)

    total_no_training_data = len(dataset_train)
    print("- Total number of training data:", total_no_training_data)


    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"----- NaN detected in parameter: {name}")

    # torch.cuda.set_device(rank)
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    model = model.to(rank)  # Move model to the appropriate GPU
    model = DDP(model, device_ids=[rank])  # Wrap with DDP

    # criterion = LossFunctionSomaSegmentation(dice_weight=0.8, focal_weight=2)
    criterion = LossFunctionSomaSegmentation(dice_weight=1, focal_weight=2)

    criterion = criterion.to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_start, weight_decay=L2_weight)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if optimizer_load_path is not None and optimizer_load_path != 'None':
        ##### FSDP
        # full_osd = torch.load(optimizer_load_path, map_location='cpu', weights_only=True)
        # sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model) 
        # optimizer.load_state_dict(sharded_osd)

        ##### DDP
        optimizer_state_dict = torch.load(optimizer_load_path, map_location='cpu')  # Load the full optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        print(f'- Optimizer loaded from {optimizer_load_path}')

    if scaler_load_path is not None and scaler_load_path != 'None': 
        scaler.load_state_dict(torch.load(scaler_load_path, map_location='cpu', weights_only=True))
        print(f'- Scaler loaded from {scaler_load_path}')
    
    # scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=scheulder_warmup_epochs, max_epochs=num_epochs, warmup_start_lr=1e-4, eta_min=learning_rate_end, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=lr_factor, min_lr=learning_rate_end, verbose=True)

    
    if train_bool:
        for epoch in tqdm(range(epoch_start, num_epochs)):

            total_loss_train = train_single_epoch_soma(model=model,
                                            train_loader=train_loader,
                                            label_mode=label_mode,
                                            rank=rank,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            accumulation_steps=accumulation_steps,
                                            L1_weight=L1_weight,
                                            L2_weight=L2_weight,
                                            FP_weight=FP_weight,
                                            scaler=scaler,
                                            sampler=sampler_train,
                                            run=run,
                                            use_amp=use_amp)
            scheduler.step(total_loss_train)


            save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
            test_single_epoch_soma(model=model, 
                test_loader=test_loader,
                label_mode=label_mode,
                rank=rank,
                criterion=criterion,
                epoch=epoch,
                dataset_test=dataset_test,
                model_save_path=model_save_path,
                smaple_test_image_freq=smaple_test_image_freq,
                test_result_save_freq=test_result_save_freq,
                run=run)


            if run is not None:
                run["train/lr"].log(scheduler.get_last_lr()[0])
                if total_loss_train is not None:
                    if torch.isnan(total_loss_train) or torch.isinf(total_loss_train):
                        print(f'Loss is NaN or Inf. Skipping epoch {epoch}')
                    else:
                        run["train/loss"].log(total_loss_train)

            if epoch%save_model_freq==0:
                print(f'- Saving model at epoch {epoch}')
                save_model(model, model_save_path, epoch, optimizer, scaler, label_mode, rank)


        print("Training complete.")
        if run is not None and rank == 0:
            run.stop()

        cleanup()

@cli.command('train_cell_segmentation')
@click.option('--encoder_load_path', default=None, help='Path to the encoder model', required=False, type=str)
@click.option('--decoder_load_path', default=None, help='Path to the decoder model', required=False, type=str)
@click.option('--optimizer_load_path', default=None, help='Path to the optimizer model', required=False, type=str)
@click.option('--scaler_load_path', default=None, help='Path to the scaler model', required=False, type=str)
@click.option('--prompt_encoder_load_path', default=None, help='Path to the prompt encoder model', required=False, type=str)
@click.option('--skip_connection_block_load_path', default=None, help='Path to the skip connection block model', required=False, type=str)
@click.option('--freeze_encoder', default=False, help='Freeze the encoder', required=True, type=bool)
@click.option('--freeze_prompt_encoder', default=False, help='Freeze the prompt encoder', required=True, type=bool)
@click.option('--freeze_skip_connection_block', default=False, help='Freeze the skip connection block', required=True, type=bool)
@click.option('--freeze_decoder', default=False, help='Freeze the decoder', required=True, type=bool)
@click.option('--task_mode', default='segmentation', help='Task mode', required=True, type=str)
@click.pass_context
def train_cell_segmentation(ctx, encoder_load_path, decoder_load_path, optimizer_load_path, scaler_load_path, prompt_encoder_load_path, skip_connection_block_load_path, freeze_encoder, freeze_prompt_encoder, freeze_skip_connection_block, freeze_decoder, task_mode):
    ctx = click.get_current_context()
    image_input_size = ctx.obj['image_input_size']
    encoder_patch_size = ctx.obj['encoder_patch_size']
    encoder_emb_dim = ctx.obj['encoder_emb_dim']
    encoder_depth = ctx.obj['encoder_depth']
    encoder_mlp_ratio = ctx.obj['encoder_mlp_ratio']
    encoder_transformer_num_heads = ctx.obj['encoder_transformer_num_heads']
    skip_connection_linear_proj_dim = ctx.obj['skip_connection_linear_proj_dim']
    residual_block_num_conv_layers = ctx.obj['residual_block_num_conv_layers']
    decoder_up_sampling_dim = ctx.obj['decoder_up_sampling_dim']
    model_save_path = ctx.obj['model_save_path']
    datasets_path_train = ctx.obj['datasets_path_train']
    datasets_path_test = ctx.obj['datasets_path_test']
    train_test_ratio = ctx.obj['train_test_ratio']
    train_bool = ctx.obj['train_bool']
    transform_bool = ctx.obj['transform_bool']
    learning_rate_start = ctx.obj['learning_rate_start']
    learning_rate_end = ctx.obj['learning_rate_end']
    patience_epochs = ctx.obj['patience_epochs']
    lr_factor = ctx.obj['lr_factor']
    batch_size = ctx.obj['batch_size']
    num_epochs = ctx.obj['num_epochs']
    accumulation_steps = ctx.obj['accumulation_steps']
    load2ram = ctx.obj['load2ram']
    smaple_test_image_freq = ctx.obj['smaple_test_image_freq']
    device = ctx.obj['device']
    neptune_project = ctx.obj['neptune_project']
    model_name = ctx.obj['model_name']
    test_result_save_freq = ctx.obj['test_result_save_freq']
    epoch_start = ctx.obj['epoch_start']
    save_model_freq = ctx.obj['save_model_freq']
    rank = ctx.obj['rank']
    world_size = ctx.obj['world_size']
    global_rank = ctx.obj['global_rank']
    L1_weight = ctx.obj['l1_weight']
    L2_weight = ctx.obj['l2_weight']
    scheulder_warmup_epochs = ctx.obj['scheulder_warmup_epochs']

    setup(rank, world_size)

    label_mode = 'branch'

    FP_weight = 0 #1e-8

    if train_bool and global_rank == 0:

        run = neptune.init_run(
            project=neptune_project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            source_files=["train.py", "models.py", "VisTrans.py"]
        )

        run["parameters"] = {
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": learning_rate_end,
            "patience_epochs": patience_epochs,
            "lr_factor": lr_factor,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "load2ram": load2ram,
            "model_save_path": model_save_path,
            "transform_bool": transform_bool,
            "accumulation_steps": accumulation_steps,
            "train_bool": train_bool,
            "datasets_path_train": datasets_path_train,
            "datasets_path_test": datasets_path_test,
            "L1_weight": L1_weight,
            "L2_weight": L2_weight,
            "FP_weight": FP_weight,
            "smaple_test_image_freq": smaple_test_image_freq,
            "device": device,
            "label_mode": label_mode,
            "model_name": model_name,
            "encoder_load_path": encoder_load_path,
            "decoder_load_path": decoder_load_path,
            "optimizer_load_path": optimizer_load_path,
            "test_result_save_freq": test_result_save_freq,
            "freeze_encoder": freeze_encoder,
            "freeze_decoder": freeze_decoder,
            "train_test_ratio": train_test_ratio,
            "epoch_start": epoch_start,
            "save_model_freq": save_model_freq,
            "encoder_patch_size": encoder_patch_size,
            "encoder_emb_dim": encoder_emb_dim,
            "encoder_depth": encoder_depth,
            "encoder_mlp_ratio": encoder_mlp_ratio,
            "skip_connection_block_load_path": skip_connection_block_load_path,
            "freeze_skip_connection_block": freeze_skip_connection_block,
            "residual_block_num_conv_layers": residual_block_num_conv_layers,
            "skip_connection_linear_proj_dim": skip_connection_linear_proj_dim,
            "decoder_up_sampling_dim": decoder_up_sampling_dim,
            "scheulder_warmup_epochs": scheulder_warmup_epochs
        }

    else:
        run = None

    spatial_transforms = {
    # # Your other random transforms here
    # # tio.RandomAffine(...): 0.7,
    # # tio.RandomFlip(...): 0.2,
    tio.RandomAffine(degrees=(-10,10,-10,10,-4,4), scales=(0.7,1.3, 0.7,1.3, 0.7,1.3), 
                     translation=(-0.5,0.5,-0.5,0.5,-0.5,0.5), center='image', 
                     include={'soma', 'image_noisy', 'trace'}, label_keys=['soma', 'trace'], 
                     image_interpolation='bspline', label_interpolation='label_gaussian'): 0.4,
    tio.RandomFlip(axes=['Left', 'right'], include={'image_noisy', 'soma', 'trace'}): 0.2,

    # Existing transforms
    tio.RandomNoise(std=(0, 0.002), include={'image_noisy', 'image'}): 0.1,
    # Gentle intensity transforms
    tio.RandomBlur(std=(0, 0.5), include={'image_noisy'}): 0.05,  # Minimal blurring
    tio.RandomGamma(log_gamma=(-0.1, 0.2), include={'image_noisy'}): 0.05,  # Minimal gamma adjustment

    
    # Careful spatial transforms
    tio.RandomElasticDeformation(max_displacement=(0.4, 0.4, 0.4), include={'soma', 'image_noisy', 'trace'}, label_keys=['soma', 'trace'], 
                     image_interpolation='bspline', label_interpolation='label_gaussian'): 0.2,  # Minimal elastic deformation
    }

    if transform_bool:
        transforms = [
            # Deterministic resize - always applied
            # tio.Resize((128, 128, 16), 
            #         include={'trace', 'image', 'image_noisy'}, 
            #         image_interpolation='linear'),
            # # Random transforms - applied with probability
            tio.OneOf(spatial_transforms, p=1)
        ]
        transform = tio.Compose(transforms)
    else:
        transform = None

    print(f'- Global rank {global_rank} is training')
    print('- Loading datasets')
    train_loader, test_loader, dataset_train, dataset_test, sampler_train, _ = make_dataloaders(datasets_path_train, 
                                                                                datasets_path_test, 
                                                                                train_test_ratio, 
                                                                                batch_size, 
                                                                                load2ram, 
                                                                                transform, 
                                                                                world_size=world_size, 
                                                                                rank=rank,
                                                                                label_mode=label_mode)
    # check_datasets(train_loader, test_loader)

    # Model Definition
    encoder = VisTrans.UNet_ImageEncoderViT3D(img_size=image_input_size, 
                                            patch_size=encoder_patch_size,
                                            in_chans=1,
                                            embed_dim=encoder_emb_dim,
                                            depth=encoder_depth,
                                            num_heads=encoder_transformer_num_heads,
                                            mlp_ratio=encoder_mlp_ratio,
                                            dropout_rate=0.05,
                                            device=rank)

    num_patches = encoder.num_patches
    prompt_encoder = encoders.PromptEncoder(embed_dim=encoder_emb_dim,
                                                input_image_size=image_input_size)


    skip_connection_block = models.SkipConnectionBlockWithCrossAttention(embedding_dim=encoder_emb_dim,
                                                                            num_heads=encoder_transformer_num_heads,
                                                                            mlp_dim=int(encoder_mlp_ratio*encoder_emb_dim),
                                                                            input_img_size=image_input_size,
                                                                            num_patches=num_patches,
                                                                            linear_proj_dim=skip_connection_linear_proj_dim,
                                                                            up_sampling_dim=decoder_up_sampling_dim)
    decoder = models.U_Net_decoder(skip_connections=True, 
                                        num_patches=num_patches,
                                        up_sampling_dim=decoder_up_sampling_dim,
                                        num_conv_layers=residual_block_num_conv_layers)

    # skip_connection_block = models.SkipConnectionBlockTransformerBasedWithCrossAttention(num_patches=num_patches, 
    #                                                                                      embed_dim=encoder_emb_dim,
    #                                                                                      num_heads=encoder_transformer_num_heads,
    #                                                                                      mlp_ratio=encoder_mlp_ratio,
    #                                                                                      depth=4)

    # decoder = models.U_Net_decoder_TransformerBased(embed_dim=encoder_emb_dim,
    #                                                 num_heads=encoder_transformer_num_heads,
    #                                                 mlp_ratio=encoder_mlp_ratio,
    #                                                 num_patches=num_patches,
    #                                                 depth=4)

    
    if encoder_load_path is not None and encoder_load_path != 'None':
        encoder = load_model(encoder, torch.load(encoder_load_path, map_location='cpu', weights_only=True))

    if decoder_load_path is not None and decoder_load_path != 'None':
        decoder = load_model(decoder, torch.load(decoder_load_path, map_location='cpu', weights_only=True))

    if skip_connection_block_load_path is not None and skip_connection_block_load_path != 'None':
        skip_connection_block = load_model(skip_connection_block, torch.load(skip_connection_block_load_path, map_location='cpu', weights_only=True))

    if prompt_encoder_load_path is not None and prompt_encoder_load_path != 'None':

        prompt_encoder = load_model(prompt_encoder, torch.load(prompt_encoder_load_path, map_location='cpu', weights_only=True))

    model = eval(f'models.{model_name}')(encoder, decoder, prompt_encoder=prompt_encoder, 
                                         skip_connection_block=skip_connection_block,
                                        freeze_encoder=freeze_encoder,
                                        freeze_prompt_encoder=freeze_prompt_encoder,
                                        freeze_decoder=freeze_decoder,
                                        freeze_skip_connection_block=freeze_skip_connection_block)


    model_total_params = sum(p.numel() for p in model.parameters())
    print("- Total parameters:", model_total_params)

    encoder_total_params = sum(p.numel() for p in encoder.parameters())
    print("- Encoder parameters:", encoder_total_params)

    decoder_total_params = sum(p.numel() for p in decoder.parameters())
    print("- Decoder parameters:", decoder_total_params)

    skip_connection_block_total_params = sum(p.numel() for p in skip_connection_block.parameters())
    print("- Skip connection block parameters:", skip_connection_block_total_params)

    prompt_encoder_total_params = sum(p.numel() for p in prompt_encoder.parameters())
    print("- Prompt encoder parameters:", prompt_encoder_total_params)

    total_no_training_data = len(dataset_train)
    print("- Total number of training data:", total_no_training_data)

    model = model.to(rank)  # Move model to the appropriate GPU
    model = DDP(model, device_ids=[rank])  # Wrap with DDP

    # criterion = LossFunctionSomaSegmentation(dice_weight=0.8, focal_weight=2)
    criterion = LossFunctionBranchSegmentation(dice_weight=1, focal_weight=4)

    criterion = criterion.to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_start, weight_decay=L2_weight)
    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if optimizer_load_path is not None and optimizer_load_path != 'None':
        ##### FSDP
        # full_osd = torch.load(optimizer_load_path, map_location='cpu', weights_only=True)
        # sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model) 
        # optimizer.load_state_dict(sharded_osd)

        ##### DDP
        optimizer_state_dict = torch.load(optimizer_load_path, map_location='cpu')  # Load the full optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        print(f'- Optimizer loaded from {optimizer_load_path}')

    if scaler_load_path is not None and scaler_load_path != 'None': 
        scaler.load_state_dict(torch.load(scaler_load_path, map_location='cpu', weights_only=True))
        print(f'- Scaler loaded from {scaler_load_path}')
    
    # scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=scheulder_warmup_epochs, max_epochs=num_epochs, warmup_start_lr=1e-4, eta_min=learning_rate_end, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=lr_factor, min_lr=learning_rate_end, verbose=True)

    
    if train_bool:
        for epoch in tqdm(range(epoch_start, num_epochs)):


            if epoch%test_result_save_freq==0:
                save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
                test_single_epoch_branch(model=model, 
                    test_loader=test_loader,
                    label_mode=label_mode,
                    rank=rank,
                    criterion=criterion,
                    epoch=epoch,
                    dataset_test=dataset_test,
                    model_save_path=model_save_path,
                    smaple_test_image_freq=smaple_test_image_freq,
                    test_result_save_freq=test_result_save_freq,
                    run=run)


            total_loss_train = train_single_epoch_branch(model=model,
                                            train_loader=train_loader,
                                            label_mode=label_mode,
                                            rank=rank,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            accumulation_steps=accumulation_steps,
                                            L1_weight=L1_weight,
                                            L2_weight=L2_weight,
                                            FP_weight=FP_weight,
                                            scaler=scaler,
                                            sampler=sampler_train,
                                            run=run,
                                            use_amp=use_amp)
            scheduler.step(total_loss_train)

            

            if run is not None:
                run["train/lr"].log(scheduler.get_last_lr()[0])
                if total_loss_train is not None:
                    if torch.isnan(total_loss_train) or torch.isinf(total_loss_train):
                        print(f'Loss is NaN or Inf. Skipping epoch {epoch}')
                    else:
                        run["train/loss"].log(total_loss_train)

            if epoch%save_model_freq==0:
                print(f'- Saving model at epoch {epoch}')
                save_model(model, model_save_path, epoch, optimizer, scaler, label_mode, rank)


        print("Training complete.")
        if run is not None and rank == 0:
            run.stop()

        cleanup()


@cli.command('train_soma_segmentation_pure_unet')
@click.option('--encoder_load_path', default=None, help='Path to the encoder model', required=False, type=str)
@click.option('--decoder_load_path', default=None, help='Path to the decoder model', required=False, type=str)
@click.option('--skip_connection_block_load_path', default=None, help='Path to the skip connection block model', required=False, type=str)
@click.option('--optimizer_load_path', default=None, help='Path to the optimizer model', required=False, type=str)
@click.option('--scaler_load_path', default=None, help='Path to the scaler model', required=False, type=str)
@click.option('--freeze_encoder', default=False, help='Freeze the encoder', required=True, type=bool)
@click.option('--freeze_skip_connection_block', default=False, help='Freeze the skip connection block', required=True, type=bool)
@click.option('--freeze_decoder', default=False, help='Freeze the decoder', required=True, type=bool)
@click.option('--task_mode', default='segmentation', help='Task mode', required=True, type=str)
@click.pass_context
def train_soma_segmentation_pure_unet(ctx, encoder_load_path, decoder_load_path, skip_connection_block_load_path, optimizer_load_path, scaler_load_path, freeze_encoder, freeze_skip_connection_block, freeze_decoder, task_mode):
    ctx = click.get_current_context()
    image_input_size = ctx.obj['image_input_size']
    encoder_patch_size = ctx.obj['encoder_patch_size']
    encoder_emb_dim = ctx.obj['encoder_emb_dim']
    encoder_depth = ctx.obj['encoder_depth']
    encoder_mlp_ratio = ctx.obj['encoder_mlp_ratio']
    encoder_transformer_num_heads = ctx.obj['encoder_transformer_num_heads']
    skip_connection_linear_proj_dim = ctx.obj['skip_connection_linear_proj_dim']
    residual_block_num_conv_layers = ctx.obj['residual_block_num_conv_layers']
    decoder_up_sampling_dim = ctx.obj['decoder_up_sampling_dim']
    model_save_path = ctx.obj['model_save_path']
    datasets_path_train = ctx.obj['datasets_path_train']
    datasets_path_test = ctx.obj['datasets_path_test']
    train_test_ratio = ctx.obj['train_test_ratio']
    train_bool = ctx.obj['train_bool']
    transform_bool = ctx.obj['transform_bool']
    learning_rate_start = ctx.obj['learning_rate_start']
    learning_rate_end = ctx.obj['learning_rate_end']
    patience_epochs = ctx.obj['patience_epochs']
    lr_factor = ctx.obj['lr_factor']
    batch_size = ctx.obj['batch_size']
    num_epochs = ctx.obj['num_epochs']
    accumulation_steps = ctx.obj['accumulation_steps']
    load2ram = ctx.obj['load2ram']
    smaple_test_image_freq = ctx.obj['smaple_test_image_freq']
    device = ctx.obj['device']
    neptune_project = ctx.obj['neptune_project']
    model_name = ctx.obj['model_name']
    test_result_save_freq = ctx.obj['test_result_save_freq']
    epoch_start = ctx.obj['epoch_start']
    save_model_freq = ctx.obj['save_model_freq']
    rank = ctx.obj['rank']
    world_size = ctx.obj['world_size']
    global_rank = ctx.obj['global_rank']
    L1_weight = ctx.obj['l1_weight']
    L2_weight = ctx.obj['l2_weight']
    scheulder_warmup_epochs = ctx.obj['scheulder_warmup_epochs']

    setup(rank, world_size)

    label_mode = 'soma'

    FP_weight = 0 #1e-8

    if train_bool and global_rank == 0:

        run = neptune.init_run(
            project=neptune_project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            source_files=["train.py", "models.py", "VisTrans.py"]
        )

        run["parameters"] = {
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": learning_rate_end,
            "patience_epochs": patience_epochs,
            "lr_factor": lr_factor,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "load2ram": load2ram,
            "model_save_path": model_save_path,
            "transform_bool": transform_bool,
            "accumulation_steps": accumulation_steps,
            "train_bool": train_bool,
            "datasets_path_train": datasets_path_train,
            "datasets_path_test": datasets_path_test,
            "L1_weight": L1_weight,
            "L2_weight": L2_weight,
            "FP_weight": FP_weight,
            "smaple_test_image_freq": smaple_test_image_freq,
            "device": device,
            "label_mode": label_mode,
            "model_name": model_name,
            "encoder_load_path": encoder_load_path,
            "decoder_load_path": decoder_load_path,
            "optimizer_load_path": optimizer_load_path,
            "test_result_save_freq": test_result_save_freq,
            "freeze_encoder": freeze_encoder,
            "freeze_decoder": freeze_decoder,
            "train_test_ratio": train_test_ratio,
            "epoch_start": epoch_start,
            "save_model_freq": save_model_freq,
            "encoder_patch_size": encoder_patch_size,
            "encoder_emb_dim": encoder_emb_dim,
            "encoder_depth": encoder_depth,
            "encoder_mlp_ratio": encoder_mlp_ratio,
            "encoder_transformer_num_heads": encoder_transformer_num_heads,
            "skip_connection_block_load_path": skip_connection_block_load_path,
            "freeze_skip_connection_block": freeze_skip_connection_block,
            "residual_block_num_conv_layers": residual_block_num_conv_layers,
            "skip_connection_linear_proj_dim": skip_connection_linear_proj_dim,
            "decoder_up_sampling_dim": decoder_up_sampling_dim,
            "scheulder_warmup_epochs": scheulder_warmup_epochs
        }

    else:
        run = None

    spatial_transforms = {
    # # Your other random transforms here
    # # tio.RandomAffine(...): 0.7,
    # # tio.RandomFlip(...): 0.2,
    # tio.RandomAffine(label_keys=['soma', 'image'], degrees=20, translation=(-10,10,-10,10,-10,10), center='origin', include={'soma', 'image'}, image_interpolation='bspline', label_interpolation='label_gaussian'): 1,
    # tio.RandomFlip(axes=['Left', 'right'], include={'image', 'soma'}): 1,
    # tio.RandomNoise(std=(0, 0.002), include={'image'}): 1,

    # Existing transforms
    tio.RandomAffine(label_keys=['soma'], 
                     degrees=10, 
                     translation=(-10,10,-10,10,-10,10), 
                     center='origin', 
                     include={'soma', 'image', 'image_noisy'}, 
                     image_interpolation='bspline', 
                     label_interpolation='label_gaussian'): 1,
    tio.RandomFlip(axes=['Left', 'right'], include={'image', 'soma', 'image_noisy'}): 1,
    # tio.RandomNoise(std=(0, 0.002), include={'image_noisy'}): 1,
    # # Gentle intensity transforms
    # tio.RandomBlur(std=(0, 0.5), include={'image', 'image_noisy'}): 0.5,  # Minimal blurring
    # tio.RandomGamma(log_gamma=(-0.1, 0.1), include={'image', 'image_noisy'}): 0.5,  # Minimal gamma adjustment
    
    # Careful spatial transforms
    # tio.RandomElasticDeformation(max_displacement=(5, 5, 5), include={'image', 'soma'}): 0.3,  # Minimal elastic deformation
    }

    if transform_bool:
        transforms = [
            # Deterministic resize - always applied
            tio.Resize((128, 128, 16), 
                    include={'soma', 'image', 'image_noisy'}, 
                    image_interpolation='linear'),
            # Random transforms - applied with probability
            tio.OneOf(spatial_transforms, p=1)
        ]
        transform = tio.Compose(transforms)
    else:
        transform = None

    print(f'- Global rank {global_rank} is training')
    print('- Loading datasets')
    train_loader, test_loader, dataset_train, dataset_test, sampler_train, _ = make_dataloaders(datasets_path_train, 
                                                                                datasets_path_test, 
                                                                                train_test_ratio, 
                                                                                batch_size, 
                                                                                load2ram, 
                                                                                transform, 
                                                                                world_size=world_size, 
                                                                                rank=rank,
                                                                                label_mode=label_mode)
    # check_datasets(train_loader, test_loader)

    # Model Definition
    # # encoder = VisTrans.UNet_ImageEncoderViT3D(img_size=image_input_size, 
    # #                                         patch_size=encoder_patch_size,
    # #                                         in_chans=1,
    # #                                         embed_dim=encoder_emb_dim,
    # #                                         depth=encoder_depth,
    # #                                         num_heads=encoder_transformer_num_heads,
    # #                                         mlp_ratio=encoder_mlp_ratio,
    # #                                         dropout_rate=0.1,
    # #                                         device=rank)
    
    # # num_patches = encoder.num_patches

    # # skip_connection_block = models.SkipConnectionBlock(input_img_size=image_input_size,
    # #                                                         num_patches=num_patches,
    # #                                                         linear_proj_dim=skip_connection_linear_proj_dim,
    # #                                                         up_sampling_dim=decoder_up_sampling_dim)
    
    # # decoder = models.U_Net_decoder(skip_connections=True, 
    # #                                     num_patches=num_patches,
    # #                                     up_sampling_dim=decoder_up_sampling_dim)

    
    # # if encoder_load_path is not None and encoder_load_path != 'None':
    # #     encoder = load_model(encoder, torch.load(encoder_load_path, map_location='cpu', weights_only=True))

    # # if decoder_load_path is not None and decoder_load_path != 'None':
    # #     decoder = load_model(decoder, torch.load(decoder_load_path, map_location='cpu', weights_only=True))

    # # if skip_connection_block_load_path is not None and skip_connection_block_load_path != 'None':
    # #     skip_connection_block = load_model(skip_connection_block, torch.load(skip_connection_block_load_path, map_location='cpu', weights_only=True))


    # model = eval(f'models.{model_name}')(encoder, decoder, skip_connection_block=skip_connection_block,
    #                         freeze_encoder=freeze_encoder, 
    #                         freeze_skip_connection_block=freeze_skip_connection_block, 
    #                         freeze_decoder=freeze_decoder)

    model = models.CellSomaSegmentationModel()
    # model = eval(f'models.{model_name}')(encoder, decoder)


    model_total_params = sum(p.numel() for p in model.parameters())
    print("- Total parameters:", model_total_params)

    total_no_training_data = len(dataset_train)
    print("- Total number of training data:", total_no_training_data)


    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"----- NaN detected in parameter: {name}")

    # torch.cuda.set_device(rank)
    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    model = model.to(rank)  # Move model to the appropriate GPU
    model = DDP(model, device_ids=[rank])  # Wrap with DDP

    # criterion = LossFunctionSomaSegmentation(dice_weight=0.8, focal_weight=2)
    criterion = LossFunctionSomaSegmentation(dice_weight=1, focal_weight=2)

    criterion = criterion.to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_start, weight_decay=L2_weight)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if optimizer_load_path is not None and optimizer_load_path != 'None':
        ##### FSDP
        # full_osd = torch.load(optimizer_load_path, map_location='cpu', weights_only=True)
        # sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model) 
        # optimizer.load_state_dict(sharded_osd)

        ##### DDP
        optimizer_state_dict = torch.load(optimizer_load_path, map_location='cpu')  # Load the full optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        print(f'- Optimizer loaded from {optimizer_load_path}')

    if scaler_load_path is not None and scaler_load_path != 'None': 
        scaler.load_state_dict(torch.load(scaler_load_path, map_location='cpu', weights_only=True))
        print(f'- Scaler loaded from {scaler_load_path}')
    
    # scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=scheulder_warmup_epochs, max_epochs=num_epochs, warmup_start_lr=1e-4, eta_min=learning_rate_end, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=lr_factor, min_lr=learning_rate_end, verbose=True)

    
    if train_bool:
        for epoch in tqdm(range(epoch_start, num_epochs)):

            if epoch%2 == 0:
                save_neptune_images(model, dataset_test, rank, run, label_mode, epoch, model_save_path)
                test_single_epoch_soma(model=model, 
                    test_loader=test_loader,
                    label_mode=label_mode,
                    rank=rank,
                    criterion=criterion,
                    epoch=epoch,
                    dataset_test=dataset_test,
                    model_save_path=model_save_path,
                    smaple_test_image_freq=smaple_test_image_freq,
                    test_result_save_freq=test_result_save_freq,
                    run=run)

            total_loss_train = train_single_epoch_soma(model=model,
                                            train_loader=train_loader,
                                            label_mode=label_mode,
                                            rank=rank,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            accumulation_steps=accumulation_steps,
                                            L1_weight=L1_weight,
                                            L2_weight=L2_weight,
                                            FP_weight=FP_weight,
                                            scaler=scaler,
                                            sampler=sampler_train,
                                            run=run,
                                            use_amp=use_amp)
            scheduler.step(total_loss_train)


            if run is not None:
                run["train/lr"].log(scheduler.get_last_lr()[0])
                if total_loss_train is not None:
                    if torch.isnan(total_loss_train) or torch.isinf(total_loss_train):
                        print(f'Loss is NaN or Inf. Skipping epoch {epoch}')
                    else:
                        run["train/loss"].log(total_loss_train)

            if epoch%save_model_freq==0:
                print(f'- Saving model at epoch {epoch}')
                if rank == 0:  # Save only on rank 0
                    print(f"Saving the model at epoch {epoch} on rank {rank}")

                    # Create save directories
                    for component in ['model', 'optimizer', 'scaler']:
                        os.makedirs(os.path.join(model_save_path, component), exist_ok=True)

                    print("Directories created")

                    # Save model components
                    model_state_dict = model.state_dict()  # Access underlying model with `model.module`

                    # Save model state dictionaries
                    torch.save(model_state_dict, os.path.join(model_save_path, 'model', f"model_{epoch}.pth"))
                    
                
                    print("Model components saved")

                    # Save optimizer state
                    torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'optimizer', f"optimizer_{epoch}.pth"))
                    print("Optimizer saved")

                    # Save scaler state
                    torch.save(scaler.state_dict(), os.path.join(model_save_path, 'scaler', f"scaler_{epoch}.pth"))
                    print("Scaler saved")

                    print("Model saving completed")


        print("Training complete.")
        if run is not None and rank == 0:
            run.stop()

        cleanup()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    cli()

