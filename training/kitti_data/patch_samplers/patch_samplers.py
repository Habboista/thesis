import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F

from .abstract_patch_sampler import PatchSampler
from ..point_cloud import PointCloud
from timethis import timethis

class NoPatchSampler(PatchSampler):
    def __init__(self):
        pass

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_shape: tuple[int, int] = (image.shape[-2], image.shape[-1])
        depth_map: Tensor = torch.from_numpy(point_cloud.to_depth_map(image_shape))
        valid_mask: Tensor = (depth_map > 0)

        return image[None], depth_map[None, None], valid_mask[None, None], torch.zeros(1, 1, *image_shape)


class EigenPatchSampler(PatchSampler):
    """Reproduces Eigen augmentations, except for color jittering which is in the Augmenter class"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)
    @timethis
    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # project point cloud to image
        shape: tuple[int, int] = (image.shape[-2], image.shape[-1])
        depth_map: Tensor = torch.from_numpy(point_cloud.to_depth_map(shape)).unsqueeze(0)

        valid_mask: Tensor = (depth_map > 0)

        # Generate crops
        image_crops: list[Tensor] = []
        depth_map_crops: list[Tensor] = []
        valid_mask_crops: list[Tensor] = []
        for _ in range(self.batch_size):
            i = random.randint(0, image.shape[-2] - self.crop_size[0])
            j = random.randint(0, image.shape[-1] - self.crop_size[1])
            
            image_crops.append(F.crop(image, i, j, *self.crop_size))
            depth_map_crops.append(F.crop(depth_map, i, j, *self.crop_size))
            valid_mask_crops.append(F.crop(valid_mask, i, j, *self.crop_size))

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)
        valid_mask = torch.stack(valid_mask_crops)

        return image, depth_map, valid_mask, torch.zeros(self.batch_size, self.batch_size, *self.crop_size)
