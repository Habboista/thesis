import random

import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._patch_sampler_abstract import PatchSampler
from ..transforms import cloud2depth
from ..transforms import scale_through_camera
from ..transforms import scale_through_depth
from ..transforms import horizontal_flip_through_camera
from timethis import timethis

    
class EigenPatchSampler(PatchSampler):
    """Reproduces Eigen augmentations, except for color jittering which is in the Augmenter class"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)
        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

    @timethis
    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        
        # Color jittering
        image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if flip:
            image, camera_parameters = horizontal_flip_through_camera(image, camera_parameters)

        # Scaling
        s = random.uniform(1., 1.2)
        image, camera_parameters = scale_through_depth(image, camera_parameters, s)
        image, camera_parameters = scale_through_camera(image, camera_parameters, 0.5)

        depth_map: Tensor = cloud2depth(point_cloud, camera_parameters)

        # Generate crops
        ii = random.choices(range(image.shape[-2] - self.crop_size[0] + 1), k=self.batch_size)
        jj = random.choices(range(image.shape[-1] - self.crop_size[1] + 1), k=self.batch_size)

        image_crops = [F.crop(image, i, j, *self.crop_size) for i, j in zip(ii, jj)]
        depth_map_crops = [F.crop(depth_map, i, j, *self.crop_size) for i, j in zip(ii, jj)]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return image, depth_map, camera_parameters