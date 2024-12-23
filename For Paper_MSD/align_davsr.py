import nibabel as nib
import os
import numpy as np
HU_MIN = -1024
HU_MAX = 3071

out_dir = 'C:/Results/CT_slice_interp/For Paper_MSD/DA-VSR'
gt_dir = 'C:/Datasets/MSD/dataset_1mm'

case_ids = os.listdir(out_dir)
case_ids.reverse()
# case_ids = [case for case in case_ids if case.endswith('.nii.gz')]

case_list = []
psnr_list = []
ssim_list = []

for case in case_ids:

    if 'test' in case:
        continue

    case_id_only = case.replace('.nii.gz', '', 1)
    print(case_id_only)

    
    output_img = nib.load(os.path.join(out_dir, case))
    out = output_img.get_fdata()
    out = np.transpose(out)
    
    gt_img = nib.load(os.path.join(gt_dir, case_id_only, f'imaging_1mm.nii.gz'))
    gt = gt_img.get_fdata()

    if case_id_only == "colon_212":
        out = out[..., :456]
    if case_id_only == "liver_187":
        continue

    out = np.clip(out, HU_MIN, HU_MAX)
    out = (out - HU_MIN) / (HU_MAX - HU_MIN)

    nib.save(nib.Nifti1Image(out, affine=gt_img.affine), os.path.join(out_dir, f'{case_id_only}_test.nii.gz'))
    pass
