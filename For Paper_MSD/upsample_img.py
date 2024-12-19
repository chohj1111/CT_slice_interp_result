import os
import numpy as np
import nibabel as nib
import skimage.transform as skitr


HU_MIN = -1024
HU_MAX = 3071


if __name__ == '__main__':
    data_dir = 'C:/Datasets/MSD/dataset_5mm'
    hr_dir = 'C:/Datasets/MSD/dataset_1mm'
    out_dir = 'Bicubic'
    case_list = ['pancreas_414']
    for case in case_list:
        img_path = os.path.join(data_dir, case, 'imaging_5mm.nii.gz')
        hr_img_path = os.path.join(hr_dir, case, 'imaging_1mm.nii.gz')
        
        img = nib.load(img_path)
        # affine = img.affine
        img_data = img.get_fdata()

        hr_img = nib.load(hr_img_path)
        hr_img_data = hr_img.get_fdata()
        affine = hr_img.affine
    
        scale = 5.0
        upsampled_img = []
        for i in range(img_data.shape[0]):
            sag_slice = img_data[i, :, :]
            upsampled_sag_slice = skitr.rescale(sag_slice, (1, scale), order=3, preserve_range=True)
            upsampled_sag_slice = upsampled_sag_slice[..., 2:-2]
            upsampled_img.append(upsampled_sag_slice)
        upsampled_img = np.stack(upsampled_img, axis=0)


        upsampled_img = np.transpose(upsampled_img)
        hr_img_data = np.transpose(hr_img_data)

        cor_proj = np.mean(hr_img_data, axis=(0, 2))
        sag_proj = np.mean(hr_img_data, axis=(0, 1))
        thr_cor = -500
        thr_sag = -800
        cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
        sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()

        upsampled_img = upsampled_img[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]
        upsampled_img = np.transpose(upsampled_img)
        
        upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
        upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)


        nib.save(nib.Nifti1Image(upsampled_img, affine=affine), os.path.join(out_dir, f'{case}.nii.gz'))

    print('done')