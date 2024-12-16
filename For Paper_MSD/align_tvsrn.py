import nibabel as nib
import os
import numpy as np
HU_MIN = -1024
HU_MAX = 3071

out_dir = 'C:/Results/CT_slice_interp/For Paper_MSD/TVSRN'
gt_dir = 'C:/Datasets/MSD/dataset_1mm'

case_ids = os.listdir(out_dir)

case_list = []
psnr_list = []
ssim_list = []

for case in case_ids:
    if 'test' in case or '.txt' in case:
        continue
    case_id_only = case.replace('_pre.nii.gz', '', 1)
    print(case_id_only)

    
    output_img = nib.load(os.path.join(out_dir, f'{case_id_only}_pre.nii.gz'))
    out = output_img.get_fdata()
    out = np.transpose(out)
    # out = np.rot90(out, axes=(1, 2), k=2)
    
    gt_img = nib.load(os.path.join(gt_dir, case_id_only, f'imaging_1mm.nii.gz'))
    gt = gt_img.get_fdata()
    gt = np.transpose(gt)

    if case_id_only == "colon_212":
        out = out[:456]
        gt = gt[:456]
    if case_id_only == "liver_187":
        continue
    gt = np.clip(gt, a_min=-1024, a_max=None)
    cor_proj = np.mean(gt, axis=(0, 2))
    sag_proj = np.mean(gt, axis=(0, 1))

    thr_cor = -500
    thr_sag = -800
    cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
    sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()
    out = out[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    out = np.transpose(out)

    nib.save(nib.Nifti1Image(out, affine=gt_img.affine), os.path.join(out_dir, f'{case_id_only}_test.nii.gz'))
    pass