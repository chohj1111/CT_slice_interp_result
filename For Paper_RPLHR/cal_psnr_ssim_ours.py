import numpy as np
import nibabel as nib
import skimage.metrics
import os

HU_MIN = -1024
HU_MAX = 2048

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


out_dir = 'C:/Results/CT_slice_interp/For Paper_RPLHR/Ours'
gt_dir = 'C:/Datasets/RPLHR-CT/test/1mm'

case_ids = os.listdir(out_dir)
# case_ids = [case for case in case_ids if case.endswith('.nii.gz')]

case_list = []
psnr_list = []
ssim_list = []

for case in case_ids:
    case_id_only = case.replace('.nii.gz', '', 1)
    
    output_img = nib.load(os.path.join(out_dir, case))
    out = output_img.get_fdata()
    
    gt_img = nib.load(os.path.join(gt_dir, f'{case_id_only}.nii.gz'))
    gt = gt_img.get_fdata()
    # # out += HU_MIN
    # out = np.clip(out, HU_MIN, HU_MAX)
    # out = (out - HU_MIN) / (HU_MAX - HU_MIN)
                
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
