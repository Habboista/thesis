import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F

from ._patch_sampler_abstract import PatchSampler
from ..transforms import render_depth_map
from timethis import timethis

    
class EigenPatchSampler(PatchSampler):
    """Reproduces Eigen augmentations, except for color jittering which is in the Augmenter class"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)

    @timethis
    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        
        depth_map: Tensor = render_depth_map(point_cloud, camera_parameters)

        # Generate crops
        ii = random.choices(range(image.shape[-2] - self.crop_size[0] + 1), k=self.batch_size)
        jj = random.choices(range(image.shape[-1] - self.crop_size[1] + 1), k=self.batch_size)

        image_crops = [F.crop(image, i, j, *self.crop_size) for i, j in zip(ii, jj)]
        depth_map_crops = [F.crop(depth_map, i, j, *self.crop_size) for i, j in zip(ii, jj)]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return image, depth_map, camera_parameters