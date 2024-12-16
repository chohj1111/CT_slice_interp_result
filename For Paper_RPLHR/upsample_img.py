import os
import numpy as np
import nibabel as nib
import skimage.transform as skitr

if __name__ == '__main__':
    data_dir = 'C:/Datasets/RPLHR-CT/test/5mm'
    hr_dir = 'C:/Datasets/RPLHR-CT/test/1mm'
    out_dir = 'Bicubic'
    case_list = ['CT00000239']
    for case in case_list:
        img_path = os.path.join(data_dir, f'{case}.nii.gz')
        hr_img_path = os.path.join(hr_dir, f'{case}.nii.gz')
        
        img = nib.load(img_path)
        affine = img.affine
        img_data = img.get_fdata()
        hr_img_shape = nib.load(hr_img_path).get_fdata().shape
        scale = 5.0
        upsampled_img = []
        for i in range(img_data.shape[0]):
            sag_slice = img_data[i, :, :]
            upsampled_sag_slice = skitr.rescale(sag_slice, (1, scale), order=3, preserve_range=True)
            upsampled_sag_slice = upsampled_sag_slice[..., 2:-2]
            upsampled_img.append(upsampled_sag_slice)
        upsampled_img = np.stack(upsampled_img, axis=0)
        upsampled_img = upsampled_img[..., 5:-5]
        nib.save(nib.Nifti1Image(upsampled_img, affine=affine), os.path.join(out_dir, f'{case}.nii.gz'))

    print('done')