import os
import random
import json

import torch
from torch.autograd import Variable
import torch.utils.data as Data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import numpy as np
import nibabel as nib

from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")


HU_MIN = -1024
HU_MAX = 3071

def normalize_kits(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    norm_img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    
    return norm_img 


output_dir = "TVSRN"
gt_dir = "C:/Datasets/kits23/dataset_1mm"

psnr_list = []
ssim_list = []
for case_name in os.listdir(output_dir):
    case = case_name.replace('_pre.nii.gz', '', 1)
    
    y_denorm = nib.load(os.path.join(gt_dir, case, 'imaging_down_1mm.nii.gz')).get_fdata()
    y_pre_denorm = nib.load(os.path.join(output_dir, case_name)).get_fdata()
     
     
    last_slice = y_denorm.shape[0] - (y_denorm.shape[0] - 1) % 5
    y_denorm = y_denorm[:last_slice]
    
    # TVSRN 
    y_denorm = y_denorm[5:-5]

    y_denorm = np.clip(y_denorm, a_min=-1024, a_max=None)
    cor_proj = np.mean(y_denorm, axis=(0, 2))
    sag_proj = np.mean(y_denorm, axis=(0, 1))

    thr_cor = -500
    thr_sag = -600
    cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
    sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()

    y_denorm = y_denorm[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    y_pre_denorm = y_pre_denorm[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]

    # clip and normalize
    y_denorm_norm = normalize_kits(y_denorm)
    y_pre_denorm_norm = normalize_kits(y_pre_denorm)
                
    # only extract synthesized slices
    interpolated_slice_indicator = np.array(range(y_denorm_norm.shape[0])) % 5 > 0
    y_denorm_norm = y_denorm_norm[interpolated_slice_indicator]
    y_pre_denorm_norm = y_pre_denorm_norm[interpolated_slice_indicator]

    psnr = 0.
    ssim = 0.
    for j in range(y_denorm_norm.shape[0]):
        psnr += peak_signal_noise_ratio(y_denorm_norm[j], y_pre_denorm_norm[j], data_range=1)
        ssim += structural_similarity(y_denorm_norm[j], y_pre_denorm_norm[j], data_range=1)
        
    psnr /= y_denorm_norm.shape[0]
    ssim /= y_denorm_norm.shape[0]                        

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    
    print(f"{case} psnr: {round(psnr, 4)} ssim: {round(ssim, 4)}")
    

psnr_avg_arr = np.array(psnr_list)
psnr_avg_mean = np.mean(psnr_avg_arr, axis=0)
ssim_avg_arr = np.array(ssim_list)
ssim_avg_mean = np.mean(ssim_avg_arr, axis=0)
    
print(f'psnr_avg_mean', psnr_avg_mean)
print(f'ssim_avg_mean', ssim_avg_mean)