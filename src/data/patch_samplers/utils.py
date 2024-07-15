import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as F

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

def get_blur_weight_mask(image: Tensor) -> Tensor:
    assert len(image.shape) >= 3, f"Expected image of size ... x 3 x H x W, got {image.shape}"
    assert image.shape[-3] == 3,  f"Expected image of size ... x 3 x H x W, got {image.shape}"

    shape = tuple(image.shape)
    shape = tuple(shape[:-3]) + (1,) + tuple(shape[-2:])
    mask = torch.ones(*shape, dtype=torch.float) * 0.1

    i1 = image.shape[-2] // 6
    j1 = image.shape[-1] // 6
    i2 = image.shape[-2] * 5 // 6
    j2 = image.shape[-1] * 5 // 6

    mask[..., i1:i2, j1:j2] = 0.5

    i1 = image.shape[-2] // 3
    j1 = image.shape[-1] // 3
    i2 = image.shape[-2] * 2 // 3
    j2 = image.shape[-1] * 2 // 3

    mask[..., i1:i2, j1:j2] = 1

    return mask