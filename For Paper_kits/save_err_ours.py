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

pred_dir = 'Ours'
out_dir = 'Ours_err'
for case in splits_dct['test']:

    saint_nib = nib.load(f'{pred_dir}/{case}.nii.gz')
    img_saint = saint_nib.get_fdata()

    gt_nib = nib.load(f'C:/Datasets/kits23_slice/{case}/imaging_1mm.nii.gz')
    img_gt = gt_nib.get_fdata()
    affine = gt_nib.affine

    last_idx = img_gt.shape[0] - (img_gt.shape[0] - 1) % 5
    img_gt = img_gt[:last_idx]


    # pad_width = [(img_gt.shape[0] - img_saint.shape[0], 0), (0, 0), (0, 0)]
    # img_saint = np.pad(img_saint, pad_width)
    
    # os.makedirs(os.path.join('images', 'thumbnail'), exist_ok=True)
    # upsampled_img = np.clip(img_gt, HU_MIN, HU_MAX)
    # upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

    # thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
    # img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
    # img_pil.save(os.path.join('images', 'thumbnail', f'{case}.jpg'))

    # upsampled_img = np.clip(img_saint, HU_MIN, HU_MAX)
    # upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

    # thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
    # img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
    # img_pil.save(os.path.join('images', 'thumbnail', f'{case}_SAINT.jpg'))

    # if case != 'case_00555':
    #     continue

    try:
        err = abs(img_saint - img_gt)
        gt_slice_indicator = np.array(range(img_gt.shape[0])) % 5 == 0
        err[gt_slice_indicator] = 0

        os.makedirs(os.path.join(out_dir, case), exist_ok=True)
        nib.save(nib.Nifti1Image(err.astype(np.float32), affine=affine),
                    os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))
    except:
        continue

