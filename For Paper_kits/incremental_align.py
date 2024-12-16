import nibabel as nib
import numpy as np
import os
from PIL import Image
import json 

HU_MIN = -1024 
HU_MAX = 3071
lower_bound = -58.0
upper_bound = 302.0

with open('splits.json', 'r') as file:
    splits_dct = json.load(file)

out_dir = 'Incremental_crop'

for case in splits_dct['test']:
    # case = 'case_00589'
    incremental_nib = nib.load(f'Incremental\{case}.nii.gz')
    img_incremental = incremental_nib.get_fdata()
    # img_incremental = np.rot90(img_incremental)
    img_incremental = img_incremental.transpose(2, 1, 0)

    gt_nib = nib.load(f'C:/Datasets/kits23/dataset_1mm/{case}/imaging_down_1mm.nii.gz')
    img_gt = gt_nib.get_fdata()
    affine = gt_nib.affine

    img_gt = np.clip(img_gt, a_min=-1024, a_max=None)
    cor_proj = np.mean(img_gt, axis=(0, 2))
    sag_proj = np.mean(img_gt, axis=(0, 1))

    thr_cor = -500
    thr_sag = -600
    cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
    sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()


    # os.makedirs(os.path.join('images', 'thumbnail'), exist_ok=True)
    # upsampled_img = np.clip(img_gt, HU_MIN, HU_MAX)
    # upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

    # thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
    # img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
    # img_pil.save(os.path.join('images', 'thumbnail', f'{case}.jpg'))

    # upsampled_img = np.clip(img_incremental, HU_MIN, HU_MAX)
    # upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

    # thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
    # img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
    # img_pil.save(os.path.join('images', 'thumbnail', f'{case}_incremental.jpg'))


    # img_incremental += HU_MIN
    img_incremental = img_incremental[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    img_incremental = np.clip(img_incremental, HU_MIN, HU_MAX)

    os.makedirs(os.path.join(out_dir, case), exist_ok=True)
    nib.save(nib.Nifti1Image(img_incremental.astype(np.float32), affine=affine),
                os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))

