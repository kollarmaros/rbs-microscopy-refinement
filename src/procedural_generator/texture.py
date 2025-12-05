import random

import cv2
import numpy as np
from perlin_numpy import generate_perlin_noise_2d  # https://github.com/pvigier/perlin-numpy
from sklearn import preprocessing

from utils import create_mask_by_hull, create_mask_by_shape, merge_segmentation_mask


def create_perlin_noise(noise_setup):
    min_max_scaler = preprocessing.MinMaxScaler()
    resolution_value = noise_setup['resolution_value']
    shape = noise_setup['shape']
    target_std = noise_setup['target_std']
    target_mean = noise_setup['target_mean']

    # np.random.seed(0)
    noise = generate_perlin_noise_2d((shape, shape), (resolution_value, resolution_value))
    noise = min_max_scaler.fit_transform(noise)

    scaled_mean = np.mean(noise)
    scaled_std = np.std(noise)
    noise = ((noise - scaled_mean) / scaled_std) * target_std + target_mean

    # noise = noise * 255
    noise = noise.astype(np.uint8)
    return noise


def texture_by_objects(noise_setup, input_image, input_mask=None, negative_mask=None):
    '''
    input_image - black objects on white background

    created texture - texture by shape and white background or vice versa if negative_mask
    '''
    noise_setup['target_std'] = random.randrange(noise_setup["target_std_range"][0] * 100,
                                                 noise_setup["target_std_range"][1] * 100) / 100
    noise_setup['target_mean'] = random.randrange(noise_setup["target_mean_range"][0] * 100,
                                                  noise_setup["target_mean_range"][1] * 100) / 100

    texture = create_perlin_noise(noise_setup)

    if noise_setup["mask_approach"] == "HULL":
        texture_mask = create_mask_by_hull(input_image)
    elif noise_setup["mask_approach"] == "SHAPE":
        texture_mask = create_mask_by_shape(input_image)
    elif noise_setup["mask_approach"] == "MASK":
        texture_mask = merge_segmentation_mask(input_mask)
    else:
        raise Exception(f"Unknown mask approach: {noise_setup['mask_approach']}")

    if negative_mask is not None:
        texture_mask = texture_mask * negative_mask

    if noise_setup["texture_outside"]:
        texture_mask = 1 - texture_mask
    texture_mask = cv2.GaussianBlur(texture_mask, noise_setup["mask_blur_kernel_shape"], 0)
    neg_texture_mask = 1 - texture_mask
    white_background = np.ones_like(input_image) * 255
    texture = texture * texture_mask + neg_texture_mask * white_background
    texture = texture.astype(np.uint8)

    return cv2.bitwise_and(texture, input_image)
