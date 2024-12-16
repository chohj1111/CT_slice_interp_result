import numpy as np
import nibabel as nib
import skimage.metrics
import os

HU_MIN = -1024
HU_MAX = 3071

def calculate_psnr_and_ssim(gt, out):
    psnr = 0.
    ssim = 0.
    
    if gt.shape[1] == out.shape[1] + 1:
        gt = gt[:, :-1]
    if gt.shape[1] == out.shape[1] + 2:
        gt = gt[:, :-2]
    if gt.shape[1] == out.shape[1] - 1:
        out = out[:, :-1]
    if gt.shape[2] == out.shape[2] + 1:
        gt = gt[:, :, :-1]
    if gt.shape[2] == out.shape[2] - 1:
        out = out[:, :, :-1]

    for j in range(out.shape[0]):
        psnr += skimage.metrics.peak_signal_noise_ratio(gt[j], out[j])
        ssim += skimage.metrics.structural_similarity(gt[j], out[j], data_range=1.)
    psnr_cor = psnr / out.shape[0]
    ssim_cor = ssim / out.shape[0]

    return psnr_cor, ssim_cor


out_dir = 'C:/Results/CT_slice_interp/For Paper_MSD/Ours'
gt_dir = 'C:/Datasets/MSD/dataset_1mm'


case_ids = os.listdir(out_dir)
case_ids = [case for case in case_ids if case.endswith('_test.nii.gz')]

case_list = []
psnr_list = []
ssim_list = []

for case in case_ids:
    case_id_only = case.replace('_test.nii.gz', '', 1)
    print(case_id_only)
    if case_id_only == "liver_187":
        continue

    output_img = nib.load(os.path.join(out_dir, f'{case_id_only}_test.nii.gz'))
    out = output_img.get_fdata()
    out = np.transpose(out)
    
    gt_img = nib.load(os.path.join(gt_dir, case_id_only, f'imaging_1mm.nii.gz'))
    gt = gt_img.get_fdata()
    gt = np.transpose(gt)

    if case_id_only == "colon_212":
        out = out[:456]
        gt = gt[:456]


    gt = np.clip(gt, a_min=-1024, a_max=None)
    cor_proj = np.mean(gt, axis=(0, 2))
    sag_proj = np.mean(gt, axis=(0, 1))

    thr_cor = -500
    thr_sag = -800
    cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
    sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()

    gt = gt[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
    gt = np.clip(gt, HU_MIN, HU_MAX)
    gt = (gt - HU_MIN) / (HU_MAX - HU_MIN)

    psnr_out_avg, ssim_out_avg = calculate_psnr_and_ssim(gt, out)
    case_list.append(case_id_only)
    psnr_list.append([psnr_out_avg])
    ssim_list.append([ssim_out_avg])
    print(case, psnr_out_avg, ssim_out_avg)
    
    
psnr_arr = np.array(psnr_list)
psnr_avg = np.mean(psnr_arr, axis=0)
ssim_arr = np.array(ssim_list)
ssim_avg = np.mean(ssim_arr, axis=0)

print(f'psnr_avg', psnr_avg)
print(f'ssim_avg', ssim_avg)


f = open(os.path.join('Ours', 'metrics.txt'), "w+")
f.write(f'case_ids\tPSNR\tSSIM\n')
for i in range(len(psnr_arr)):
    f.write(f'{case_list[i]}\t{psnr_arr[i][-1]:.2f}\t{ssim_arr[i][-1]:.4f}\n')
f.write(f'case_avg\t{psnr_avg[-1]:.2f}\t{ssim_avg[-1]:.4f}\n')        
f.close()
