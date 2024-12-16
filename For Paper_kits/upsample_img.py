import os
import numpy as np
import nibabel as nib
import skimage.transform as skitr

if __name__ == '__main__':
    data_dir = 'C:/Datasets/kits23/dataset_5mm'
    # data_dir = "/home/cvip/dataset/230720_kits23_test_images"
    case_list = ['case_00299']
    # case_list = ['case_00091', 'case_00202']
    for case in case_list:
        case_dir = os.path.join(data_dir, case)
        img_path = os.path.join(case_dir, 'imaging_down_5mm.nii.gz')
        
        img = nib.load(img_path)
        affine = img.affine
        img_data = img.get_fdata()
        upsampled_img = []
        scale = 5.0
        for i in range(512):
            sag_slice = img_data[:, i, :]
            upsampled_sag_slice = skitr.rescale(sag_slice, (scale, 1), order=3)
            upsampled_sag_slice = upsampled_sag_slice[2:-2]
            upsampled_img.append(upsampled_sag_slice)

        upsampled_img = np.stack(upsampled_img, axis=1)
        affine_down = affine
        affine_down[2, 0] = affine[2, 0] / scale
        nib.save(nib.Nifti1Image(upsampled_img, affine=affine_down), os.path.join(data_dir, case, f'imaging_up_1mm_bicubic_cor.nii.gz'))

    print('done')