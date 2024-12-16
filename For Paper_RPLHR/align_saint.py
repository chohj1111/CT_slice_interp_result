import nibabel as nib
import os
import numpy as np
HU_MIN = -1024
HU_MAX = 2048

out_dir = 'C:/Results/CT_slice_interp/For Paper_RPLHR/SAINT'
gt_dir = 'C:/Datasets/RPLHR-CT/test/1mm'

case_ids = os.listdir(out_dir)
# case_ids = [case for case in case_ids if case.endswith('.nii.gz')]

case_list = []
psnr_list = []
ssim_list = []

for case in case_ids:
    if 'REF' not in case:
        continue
    case_id_only = case.replace('.nii.gz_x5_REF.nii.gz', '', 1)
    
    output_img = nib.load(os.path.join(out_dir, case))
    out = output_img.get_fdata()
    
    gt_img = nib.load(os.path.join(gt_dir, f'{case_id_only}.nii.gz'))
    gt = gt_img.get_fdata()

    out = np.clip(out, HU_MIN, HU_MAX)
    out = (out - HU_MIN) / (HU_MAX - HU_MIN)
    
    out = out[..., 5:-5]
                
    nib.save(nib.Nifti1Image(out, affine=gt_img.affine), os.path.join(out_dir, f'{case_id_only}.nii.gz'))
    pass