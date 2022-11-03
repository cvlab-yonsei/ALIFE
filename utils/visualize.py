import torch
import numpy as np

from PIL import Image
from scipy.io import loadmat

ADE_COLORS = loadmat('../tmp/data/color150.mat')['colors'] # (150,3)
ADE_COLORS = np.concatenate(([[224, 224, 192]], ADE_COLORS), axis=0).astype('uint8') # (151,3)

CLASS_COLORS = (
    (0, 0, 0),
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), 
    (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), 
    (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), 
    (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    (224, 224, 192)
)


def decode_VOC(mask):
    num_classes = len(CLASS_COLORS)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    h, w = mask.shape
    
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = CLASS_COLORS[k]
            if k == 255:
                pixels[k_, j_] = CLASS_COLORS[-1]
    output = np.array(img)
    return output


def get_npimg(img):
    x = img.clone()
    if x.dim() == 3 and x.shape[0] == 1:
        return x.squeeze().detach().cpu().numpy()
        
    if x.dim() == 4:
        if x.shape[1] == 1:
            return x.squeeze().detach().cpu().numpy()
        else:
            x = x.squeeze(0)
        
    x = x.detach().cpu()
    x = x.permute(1,2,0).numpy()
    x -= x.min()
    x /= x.max() + 1e-7
    return x


def colorEncode(labelmap, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(ADE_COLORS[label], (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb
    
    
def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret