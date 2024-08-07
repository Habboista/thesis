import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as F

__all__ = [
    "blur",
    "get_blur_weight_mask",
    "clean_corner_response",
]

def blur(image: Tensor) -> Tensor:
    blur_0 = image
    blur_1 = F.gaussian_blur(blur_0, (5, 5), (1.5, 1.5))
    blur_2 = F.gaussian_blur(blur_1, (5, 5), (1.5, 1.5))
    blur_3 = F.gaussian_blur(blur_2, (5, 5), (1.5, 1.5))

    blur = blur_3
    i1 = image.shape[-2] // 6
    j1 = image.shape[-1] // 6
    i2 = image.shape[-2] * 5 // 6
    j2 = image.shape[-1] * 5 // 6
    blur[..., i1:i2, j1:j2] = blur_2[..., i1:i2, j1:j2]

    i1 = image.shape[-2] // 3
    j1 = image.shape[-1] // 3
    i2 = image.shape[-2] * 2 // 3
    j2 = image.shape[-1] * 2 // 3
    blur[..., i1:i2, j1:j2] = blur_1[..., i1:i2, j1:j2]

    i1 = int(image.shape[-2] * 2.5 // 6)
    j1 = int(image.shape[-1] * 2.5 // 6)
    i2 = int(image.shape[-2] * 4.5 // 6)
    j2 = int(image.shape[-1] * 4.5 // 6)
    blur[..., i1:i2, j1:j2] = blur_0[..., i1:i2, j1:j2]

    return blur

def get_blur_weight_mask(shape: tuple[int, int]) -> Tensor:
    mask = torch.ones(*shape, dtype=torch.float) * 0.1

    i1 = shape[-2] // 6
    j1 = shape[-1] // 6
    i2 = shape[-2] * 5 // 6
    j2 = shape[-1] * 5 // 6

    mask[..., i1:i2, j1:j2] = 0.5

    i1 = shape[-2] // 3
    j1 = shape[-1] // 3
    i2 = shape[-2] * 2 // 3
    j2 = shape[-1] * 2 // 3

    mask[..., i1:i2, j1:j2] = 1

    return mask

def clean_corner_response(np_corner_response: np.ndarray) -> np.ndarray:
    # Cancel response near image boundaries (otherwise the warp contains out-of-view points)
    # TODO: use a different shape for the allowed area than the rectangle
    p1 = 0.2
    p2 = 0.8
    i1 = int(0.4 * np_corner_response.shape[-2])
    i2 = int(p2 * np_corner_response.shape[-2])
    j1 = int(p1 * np_corner_response.shape[-1])
    j2 = int(p2 * np_corner_response.shape[-1]) 
    np_corner_response[:i1, :] = 0
    np_corner_response[i2:, :] = 0
    np_corner_response[:, :j1] = 0
    np_corner_response[:, j2:] = 0

    return np_corner_response