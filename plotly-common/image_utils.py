import plotly.express as px
import numpy as np
import itertools


def fromhex(n):
    return int(n, base=16)


import numpy as np
import itertools
from binascii import unhexlify as fromhex

def label_to_colors(img, colormap=px.colors.qualitative.Light24, alpha=128, color_class_offset=0, labels_contiguous=False, no_map_zero=False):
    """
    Take a numpy array 'img' containing label data and return a numpy array
    of shape (..., 4) where each label is mapped to an RGBA color.
    """
    # Convert color hex codes to RGB tuples
    colormap = [tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for h in colormap]
    
    if not isinstance(alpha, list):
        alpha = [alpha] * len(colormap)  # Ensure alpha is iterable
    
    cm_alpha = list(zip(colormap, itertools.cycle(alpha)))  # Combine colors with alphas

    # Prepare output image arrays
    cimg = np.zeros(img.shape + (3,), dtype="uint8")  # RGB channels
    alpha_img = np.zeros(img.shape + (1,), dtype="uint8")  # Alpha channel
    
    labels = range(img.min(), img.max() + 1) if labels_contiguous else np.unique(img)

    # Apply colors and alpha to each label
    for label in labels:
        if label == 0 and no_map_zero:
            continue
        color, alpha_value = cm_alpha[(label + color_class_offset) % len(cm_alpha)]
        mask = (img == label)
        cimg[mask] = color
        alpha_img[mask] = alpha_value

    # Combine color and alpha channels to form RGBA image
    return np.concatenate((cimg, alpha_img), axis=-1)



def combine_last_dim(
    img,
    output_n_dims=3,
    combiner=lambda x: (np.sum(np.abs(x), axis=-1) != 0).astype("float"),
):
    if len(img.shape) == output_n_dims:
        return img
    imgout = combiner(img)
    return imgout
