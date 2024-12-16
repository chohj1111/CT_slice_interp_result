import nibabel as nib
import numpy as np
import os
from PIL import Image
import json
import matplotlib.pyplot as plt


HU_MIN = -1024 
HU_MAX = 2048
lower_bound = 0.0688
upper_bound = 0.4625

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

case = 'CT00000239'
gt_dir = 'GT_align'
ours_dir = 'Ours'
saint_dir = 'SAINT'
davsr_dir = 'DA-VSR'
tvsrn_dir = 'TVSRN'
bicubic_dir = 'Bicubic'
slice_idx = 212

# gt
gt_nib = nib.load(f'{gt_dir}/{case}_test.nii.gz')
img_gt = gt_nib.get_fdata()

# ours
ours_nib = nib.load(f'{ours_dir}/{case}_test.nii.gz')
img_ours = ours_nib.get_fdata()

# saint
saint_nib = nib.load(f'{saint_dir}/{case}.nii.gz')
img_saint = saint_nib.get_fdata()
        
# DA-VSR
davsr_nib = nib.load(f'{davsr_dir}/{case}.nii.gz')
img_davsr = davsr_nib.get_fdata()

# TVSRN
tvsrn_nib = nib.load(f'{tvsrn_dir}/{case}_pre.nii.gz')
img_tvsrn = tvsrn_nib.get_fdata()

# Bicubic
bicubic_nib = nib.load(f'{bicubic_dir}/{case}.nii.gz')
img_bicubic = bicubic_nib.get_fdata()

os.makedirs(os.path.join('images', 'thumbnail'), exist_ok=True)

img_clip = np.clip(img_gt[slice_idx], lower_bound, upper_bound)
img_gt_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_gt_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_gt.jpg'))


# ours
img_clip = np.clip(img_ours[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_ours.jpg'), quality=100)

# saint
img_clip = np.clip(img_saint[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_saint.jpg'), quality=100)

# da-vsr
img_clip = np.clip(img_davsr[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_davsr.jpg'), quality=100)

# tvsrn
img_clip = np.clip(img_tvsrn[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_tvsrn.jpg'), quality=100)

# Bicubic
img_clip = np.clip(img_bicubic[slice_idx], lower_bound, upper_bound)
img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

img_pil = Image.fromarray(np.uint8(255 * img_norm))
img_pil = img_pil.rotate(90)
img_pil.save(os.path.join('images', 'thumbnail', f'{case}_bicubic.jpg'), quality=100)
