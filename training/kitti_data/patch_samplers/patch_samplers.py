import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F

from .abstract_patch_sampler import PatchSampler
from ..point_cloud import PointCloud
from timethis import timethis

@timethis
class NoPatchSampler(PatchSampler):
    def __init__(self):
        pass

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        depth_map: Tensor = point_cloud.to_depth_map()

        return image.unsqueeze(0), depth_map.unsqueeze(0)

class TestPatchSampler(PatchSampler):
    def __init__(self):
        self.crop_size = (172, 576)

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        # project point cloud to image
        depth_map: Tensor = point_cloud.to_depth_map()

        # Generate crops
        ii = random.choices(range(image.shape[-2] - self.crop_size[0] + 1), k=self.batch_size)
        jj = random.choices(range(image.shape[-1] - self.crop_size[1] + 1), k=self.batch_size)

        image_crops = [F.crop(image, i, j, *self.crop_size) for i, j in zip(ii, jj)]
        depth_map_crops = [F.crop(depth_map, i, j, *self.crop_size) for i, j in zip(ii, jj)]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return image, depth_map
    
class EigenPatchSampler(PatchSampler):
    """Reproduces Eigen augmentations, except for color jittering which is in the Augmenter class"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)

    @timethis
    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        # project point cloud to image
        depth_map: Tensor = point_cloud.to_depth_map()

        # Generate crops
        ii = random.choices(range(image.shape[-2] - self.crop_size[0] + 1), k=self.batch_size)
        jj = random.choices(range(image.shape[-1] - self.crop_size[1] + 1), k=self.batch_size)

        image_crops = [F.crop(image, i, j, *self.crop_size) for i, j in zip(ii, jj)]
        depth_map_crops = [F.crop(depth_map, i, j, *self.crop_size) for i, j in zip(ii, jj)]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return image, depth_map