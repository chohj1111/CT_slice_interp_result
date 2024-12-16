import numpy as np
import nibabel as nib
import skimage.metrics
import os

HU_MIN = -1024
HU_MAX = 3071

def calculate_psnr_and_ssim(gt, out):
    psnr = 0.
    ssim = 0.
    for j in range(out.shape[0]):
        psnr += skimage.metrics.peak_signal_noise_ratio(gt[j], out[j])
        ssim += skimage.metrics.structural_similarity(gt[j], out[j], data_range=1.)
    psnr_cor = psnr / out.shape[0]
    ssim_cor = ssim / out.shape[0]

    return psnr_cor, ssim_cor


out_dir = './'
gt_dir = 'C:/Datasets/MSD/dataset_1mm'

case_ids = os.listdir(out_dir)
case_ids = [case for case in case_ids if case.endswith('_REF.nii.gz')]


psnr_list = []
ssim_list = []

for case in case_ids:
    case_id_only = case.replace('_x5_REF.nii.gz', '', 1)
    
    output_img = nib.load(os.path.join(out_dir, case))
    out = output_img.get_fdata()
    
    gt_img = nib.load(os.path.join(gt_dir, case_id_only, 'imaging_1mm.nii.gz'))
    gt = gt_img.get_fdata()

    out = out.transpose()
    gt = gt.transpose()
    
    if case_id_only == "colon_212":
        gt = gt[:365]
        out = out[:365]
    if case_id_only == "liver_187":
        continue

    last_idx = gt.shape[0] - (gt.shape[0]) % 5
    gt = gt[:last_idx]
    # out = out[:last_idx]

    gt = np.clip(gt, a_min=HU_MIN, a_max=None)
    cor_proj = np.mean(gt, axis=(0, 2))
    sag_proj = np.mean(gt, axis=(0, 1))

    thr_cor = -500
    thr_sag = -800
    cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
    sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()

    out = out[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    gt = gt[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
   
    out += HU_MIN
    out = np.clip(out, HU_MIN, HU_MAX)
    out = (out - HU_MIN) / (HU_MAX - HU_MIN)
        
    gt = np.clip(gt, HU_MIN, HU_MAX)
    gt = (gt - HU_MIN) / (HU_MAX - HU_MIN)

    interpolated_slice_indicator = np.array(range(gt.shape[0])) % 5 > 0
    out = out[interpolated_slice_indicator]
    gt = gt[interpolated_slice_indicator]

    psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(gt, out)
    psnr_list.append([psnr_out_avg])
    ssim_list.append([ssim_out_avg])
    
    print(case, psnr_out_avg, ssim_out_avg)
    
    
psnr_arr = np.array(psnr_list)
psnr_avg = np.mean(psnr_arr, axis=0)
ssim_arr = np.array(ssim_list)
ssim_avg = np.mean(ssim_arr, axis=0)

print(f'psnr_avg', psnr_avg)
print(f'ssim_avg', ssim_avg)