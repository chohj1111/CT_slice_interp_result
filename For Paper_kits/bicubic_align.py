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

out_dir = 'Bicubic'

for case in ['case_00299']:
# for case in ['case_00091', 'case_00202']:
    bicubic_nib = nib.load(f'C:/Datasets/kits23/dataset_5mm/{case}/imaging_up_1mm_bicubic_cor.nii.gz')
    img_bicubic = bicubic_nib.get_fdata()

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

    img_bicubic = img_bicubic[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    img_bicubic = np.clip(img_bicubic, HU_MIN, HU_MAX)

    os.makedirs(os.path.join(out_dir, case), exist_ok=True)
    nib.save(nib.Nifti1Image(img_bicubic.astype(np.float32), affine=affine),
                os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))

