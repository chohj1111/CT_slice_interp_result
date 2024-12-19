import nibabel as nib
import numpy as np
import os
from PIL import Image
import json
import matplotlib.pyplot as plt


HU_MIN = -1024 
HU_MAX = 3071
lower_bound = 0.
upper_bound = 0.3463

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

case = 'pancreas_414'
ours_dir = 'Ours'
saint_dir = 'SAINT'
# incremental_dir = 'Incremental'
davsr_dir = 'DA-VSR'
tvsrn_dir = 'TVSRN'
bicubic_dir = 'Bicubic'
gt_dir = 'GT_align'
slice_idx = 139

# gt
gt_nib = nib.load(os.path.join(gt_dir, f'{case}_gt.nii.gz'))
img_gt = gt_nib.get_fdata()
img_gt = img_gt.transpose(1, 2, 0)
img_gt = np.rot90(img_gt, axes=(1, 2), k=2)

# ours
ours_nib = nib.load(f'{ours_dir}/{case}_test.nii.gz')
img_ours = ours_nib.get_fdata()
img_ours = img_ours.transpose(1, 2, 0)
img_ours = np.rot90(img_ours, axes=(1, 2), k=2)

# saint
saint_nib = nib.load(f'{saint_dir}/{case}.nii.gz')
img_saint = saint_nib.get_fdata()
img_saint = img_saint.transpose(1, 2, 0)
img_saint = np.rot90(img_saint, axes=(1, 2), k=2)
    
# # incremental
# incremental_nib = nib.load(f'{incremental_dir}/{case}/imaging_1mm.nii.gz')
# img_incremental = incremental_nib.get_fdata()
# img_saint = img_saint.transpose(1, 2, 0)
# img_saint = np.rot90(img_saint, axes=(1, 2), k=2)
    
# DA-VSR
davsr_nib = nib.load(f'{davsr_dir}/{case}.nii.gz')
img_davsr = davsr_nib.get_fdata()
img_davsr = img_davsr.transpose(1, 0, 2)
img_davsr = np.rot90(img_davsr, axes=(1, 2), k=2)
img_davsr = np.clip(img_davsr, HU_MIN, HU_MAX)
img_davsr = (img_davsr - HU_MIN) / (HU_MAX - HU_MIN)


# TVSRN
tvsrn_nib = nib.load(f'{tvsrn_dir}/{case}_test.nii.gz')
img_tvsrn = tvsrn_nib.get_fdata()
img_tvsrn = img_tvsrn.transpose(1, 2, 0)
img_tvsrn = np.rot90(img_tvsrn, axes=(1, 2), k=2)

# Bicubic
bicubic_nib = nib.load(f'{bicubic_dir}/{case}.nii.gz')
img_bicubic = bicubic_nib.get_fdata()
img_bicubic = img_bicubic.transpose(1, 2, 0)
img_bicubic = np.rot90(img_bicubic, axes=(1, 2), k=2)


# save thumbnails 
os.makedirs(os.path.join('images', 'thumbnail'), exist_ok=True)

# gt
img_clip = np.clip(img_gt[slice_idx], lower_bound, upper_bound)
img_gt_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_gt_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_gt.jpg'))

# ours
img_clip = np.clip(img_ours[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_ours.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_ours_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_ours_err.jpg'))

# saint
img_clip = np.clip(img_saint[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_saint.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_saint_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_saint_err.jpg'))

# # incremental
# img_clip = np.clip(img_incremental[slice_idx], lower_bound, upper_bound)
# img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
# img_pil = Image.fromarray(np.uint8(255 * img_norm))
# img_pil.save(os.path.join('images', 'thumbnail', f'{case}_incremental.jpg'))
# # img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# # img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_incremental_err.jpg'))
# # plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_incremental_err.jpg'))

# da-vsr
img_clip = np.clip(img_davsr[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))

# tvsrn
img_clip = np.clip(img_tvsrn[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_tvsrn.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_tvsrn_err.jpg'))

# Bicubic
img_clip = np.clip(img_bicubic[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_bicubic.jpg'))
# img_err_pil = Image.fromarray(np.uint8(255 * abs(img_norm - img_gt_norm)))
# img_err_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr_err.jpg'))
# plot_heatmap(abs(img_norm - img_gt_norm), os.path.join('images', 'thumbnail', f'{case}_bicubic_err.jpg'))
