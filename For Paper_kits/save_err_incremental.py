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

pred_dir = 'Incremental_crop'
out_dir = 'Incremental_err'

for case in splits_dct['test']:
    incremental_nib = nib.load(f'{pred_dir}/{case}/imaging_1mm.nii.gz')
    img_incremental = incremental_nib.get_fdata()

    gt_nib = nib.load(f'C:/Datasets/kits23_slice/{case}/imaging_1mm.nii.gz')
    img_gt = gt_nib.get_fdata()
    affine = gt_nib.affine
    
    last_idx = img_gt.shape[0] - (img_gt.shape[0] - 1) % 5
    img_gt = img_gt[:last_idx]
    
    try:
        err = abs(img_incremental - img_gt)
        gt_slice_indicator = np.array(range(img_gt.shape[0])) % 5 == 0
        err[gt_slice_indicator] = 0

        os.makedirs(os.path.join(out_dir, case), exist_ok=True)
        nib.save(nib.Nifti1Image(err.astype(np.float32), affine=affine),
                    os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))
    except:
        continue

