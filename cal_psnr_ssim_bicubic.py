import numpy as np
import nibabel as nib
import skimage.metrics
import os

HU_MIN = -1024
HU_MAX = 3071

def calculate_psnr_and_ssim(gt, out):
    psnr = 0.
    ssim = 0.
    psnr += skimage.metrics.peak_signal_noise_ratio(gt, out)
    ssim += skimage.metrics.structural_similarity(gt, out, data_range=1.)

    return psnr, ssim


case = 'case_00091'
ours_dir = 'Ours'
saint_dir = 'SAINT_crop'
incremental_dir = 'Incremental_crop'
davsr_dir = 'DA-VSR'
tvsrn_dir = 'TVSRN_crop'
bicubic_dir = 'Bicubic'

# gt
gt_nib = nib.load(f'C:/Datasets/kits23_slice/{case}/imaging_1mm.nii.gz')
img_gt = gt_nib.get_fdata()
last_idx = img_gt.shape[0] - (img_gt.shape[0] - 1) % 5
img_gt = img_gt[:last_idx]
img_gt = img_gt[5:-5]
img_gt = img_gt.transpose(1, 0, 2)

# ours
ours_nib = nib.load(f'{ours_dir}/{case}.nii.gz')
img_ours = ours_nib.get_fdata()
img_ours = img_ours[5:-5]
img_ours = img_ours.transpose(1, 0, 2)

# saint
saint_nib = nib.load(f'{saint_dir}/{case}/imaging_1mm.nii.gz')
img_saint = saint_nib.get_fdata()
img_saint = img_saint[:last_idx]
img_saint = img_saint[5:-5]
img_saint = img_saint.transpose(1, 0, 2)
    
# incremental
incremental_nib = nib.load(f'{incremental_dir}/{case}/imaging_1mm.nii.gz')
img_incremental = incremental_nib.get_fdata()
img_incremental = img_incremental[5:-5]
img_incremental = img_incremental.transpose(1, 0, 2)
    
# DA-VSR
davsr_nib = nib.load(f'{davsr_dir}/{case}.nii.gz')
img_davsr = davsr_nib.get_fdata()
img_davsr = img_davsr[5:-5]
img_davsr = img_davsr.transpose(1, 0, 2)

# TVSRN
tvsrn_nib = nib.load(f'{tvsrn_dir}/{case}/imaging_1mm.nii.gz')
img_tvsrn = tvsrn_nib.get_fdata()
img_tvsrn = img_tvsrn[5:-5]
img_tvsrn = img_tvsrn.transpose(1, 0, 2)

# Bicubic
bicubic_nib = nib.load(f'{bicubic_dir}/{case}/imaging_1mm.nii.gz')
img_bicubic = bicubic_nib.get_fdata()
img_bicubic = img_bicubic[5:-5]
img_bicubic = img_bicubic.transpose(1, 0, 2)


img_gt = np.clip(img_gt[109], HU_MIN, HU_MAX)
img_gt = (img_gt - HU_MIN) / (HU_MAX - HU_MIN)

img_ours = np.clip(img_ours[109], HU_MIN, HU_MAX)
img_ours = (img_ours - HU_MIN) / (HU_MAX - HU_MIN)

img_saint = np.clip(img_saint[109], HU_MIN, HU_MAX)
img_saint = (img_saint - HU_MIN) / (HU_MAX - HU_MIN)

img_incremental = np.clip(img_incremental[109], HU_MIN, HU_MAX)
img_incremental = (img_incremental - HU_MIN) / (HU_MAX - HU_MIN)

img_davsr = np.clip(img_davsr[109], HU_MIN, HU_MAX)
img_davsr = (img_davsr - HU_MIN) / (HU_MAX - HU_MIN)

img_tvsrn = np.clip(img_tvsrn[109], HU_MIN, HU_MAX)
img_tvsrn = (img_tvsrn - HU_MIN) / (HU_MAX - HU_MIN)
    
img_bicubic = np.clip(img_bicubic[109], HU_MIN, HU_MAX)
img_bicubic = (img_bicubic - HU_MIN) / (HU_MAX - HU_MIN)



psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_ours)
print('Ours:', psnr_out_avg, ssim_out_avg)

psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_saint)
print('SAINT:', psnr_out_avg, ssim_out_avg)

psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_incremental)
print('Incremental:', psnr_out_avg, ssim_out_avg)

psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_davsr)
print('DA-VSR:', psnr_out_avg, ssim_out_avg)

psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_tvsrn)
print('TVSRN:', psnr_out_avg, ssim_out_avg)

psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(img_gt, img_bicubic)
print('Bicubic:', psnr_out_avg, ssim_out_avg)
