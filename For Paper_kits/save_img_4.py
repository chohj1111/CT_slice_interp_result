import nibabel as nib
import numpy as np
import os
from PIL import Image
import json
import matplotlib.pyplot as plt


HU_MIN = -1024 
HU_MAX = 3071
lower_bound = -206
upper_bound = 400

def plot_heatmap(error_map, filename):
    array_height, array_width = error_map.shape
    dpi = 300  # You can choose a base DPI, and the figure size will be adjusted accordingly
    figsize = array_width / float(dpi), array_height / float(dpi)

    # Create figure and axes with the calculated size
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot the heatmap without axes
    ax.imshow(error_map, cmap='jet', interpolation='nearest')
    ax.axis('off')  # Hide axes

    # Adjust layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the figure with the original data dimensions
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)

case = 'case_00091'
ours_dir = 'Ours'
saint_dir = 'SAINT_crop'
incremental_dir = 'Incremental_crop'
davsr_dir = 'DA-VSR'
tvsrn_dir = 'TVSRN_crop'
bicubic_dir = 'Bicubic'

# gt
gt_nib = nib.load(f'C:/Datasets/kits23_slice/{case}/imaging_1mm.nii.gz')
img_gt = gt_nib.get_fdata()
last_idx = img_gt.shape[0] - (img_gt.shape[0] - 1) % 5
img_gt = img_gt[:last_idx]
img_gt = img_gt[5:-5]
img_gt = img_gt.transpose(2, 0, 1)
img_gt = img_gt[...,:-1]

# ours
ours_nib = nib.load(f'{ours_dir}/{case}.nii.gz')
img_ours = ours_nib.get_fdata()
img_ours = img_ours[5:-5]
img_ours = img_ours.transpose(2, 0, 1)
img_ours = img_ours[...,:-1]

# saint
saint_nib = nib.load(f'{saint_dir}/{case}/imaging_1mm.nii.gz')
img_saint = saint_nib.get_fdata()
img_saint = img_saint[:last_idx]
img_saint = img_saint[5:-5]
img_saint = img_saint.transpose(2, 0, 1)
    
# incremental
incremental_nib = nib.load(f'{incremental_dir}/{case}/imaging_1mm.nii.gz')
img_incremental = incremental_nib.get_fdata()
img_incremental = img_incremental[5:-5]
img_incremental = img_incremental.transpose(2, 0, 1)
    
# DA-VSR
davsr_nib = nib.load(f'{davsr_dir}/{case}.nii.gz')
img_davsr = davsr_nib.get_fdata()
img_davsr = img_davsr[5:-5]
img_davsr = img_davsr.transpose(2, 0, 1)
img_davsr = img_davsr[...,:-1]

# TVSRN
tvsrn_nib = nib.load(f'{tvsrn_dir}/{case}/imaging_1mm.nii.gz')
img_tvsrn = tvsrn_nib.get_fdata()
img_tvsrn = img_tvsrn[5:-5]
img_tvsrn = img_tvsrn.transpose(2, 0, 1)

# Bicubic
bicubic_nib = nib.load(f'{bicubic_dir}/{case}/imaging_1mm.nii.gz')
img_bicubic = bicubic_nib.get_fdata()
img_bicubic = img_bicubic[5:-5]
img_bicubic = img_bicubic.transpose(2, 0, 1)

os.makedirs(os.path.join('images', 'thumbnail'), exist_ok=True)

img_clip = np.clip(img_gt[128], lower_bound, upper_bound)
img_gt_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_gt_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_gt.jpg'))

img_clip = np.clip(img_ours[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

# ours
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_ours.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_ours_err.jpg'))
plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_ours_err.jpg'))

# saint
img_clip = np.clip(img_saint[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_saint.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_saint_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_saint_err.jpg'))

# incremental
img_clip = np.clip(img_incremental[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_incremental.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_incremental_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_incremental_err.jpg'))

# da-vsr
img_clip = np.clip(img_davsr[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))

# tvsrn
img_clip = np.clip(img_tvsrn[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_tvsrn.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_tvsrn_err.jpg'))

# Bicubic
img_clip = np.clip(img_bicubic[128], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_bicubic.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_bicubic_err.jpg'))
