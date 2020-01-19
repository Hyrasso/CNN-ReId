from enum import Enum
import numpy as np
import skimage

class FillMode(Enum):
    """ Enum representing the different fill modes available
    
    ZEROES: fill with 0
    MIRROR: fill with the original image, mirrored along the axis filled
    """
    ZEROES = 0
    MIRROR = 1

def resize(array: np.ndarray, target_size: tuple, keep_ratio=True, fill_mode: FillMode=FillMode.MIRROR):
    """ Resize an array of images

    array: np.array of images with shape NxHeightxWidthxChannels 
    target_size: target shape HeightxWidthxChannels
    keep_ratio: True to keep original ratio, if part of the target image is not covered, it is filled using fill_mode
    fill_mode: see FillMode enum
    """
    original_size = array.shape[1:]
    res = np.zeros((array.shape[0], *target_size))
    for i in range(len(array)):
        # upscale 1.75 -> 224, 112, 3
        if keep_ratio:
            res[i, :224, :112, :] = skimage.transform.resize(array[i], (224, 112, 3))
        else:
            res[i] = skimage.transform.resize(array[i], target_size)
    # mirror x -> 224, 224, 3
    if fill_mode == FillMode.MIRROR:
        res[:, :, 112:, :] = res[:, :, :112, :][:,:,::-1,:]

    return res
