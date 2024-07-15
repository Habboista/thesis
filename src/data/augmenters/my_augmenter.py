import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .abstract_augmenter import Augmenter
from ..point_cloud import PointCloud

class MyAugmenter(Augmenter):
    def __init__(
            self,
            color_jittering: bool,
            hflip: bool,
            scaling_range: tuple[float, float],
        ):
        self.color_jittering = color_jittering
        self.hflip = hflip
        self.scaling_range = scaling_range

        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        # Color jittering
        if self.color_jittering:
            image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if self.hflip and flip:
            point_cloud.camera_info['K'][0, 0] *= -1
            point_cloud.camera_info['K'][0, 2] = point_cloud.camera_info['im_shape'][1] - point_cloud.camera_info['K'][0, 2]
            image = F.hflip(image)
        
        # Scaling by s
        s = random.uniform(*self.scaling_range)
        point_cloud.points[:, 0] /= s

        resized_shape: tuple[int, int] = (int(image.shape[-2] * s), int(image.shape[-1] * s))
        image = F.resize(image, resized_shape)

        # Adjust image size
        point_cloud.camera_info['K'][0, 2] *= s
        point_cloud.camera_info['K'][1, 2] *= s

        point_cloud.camera_info['im_shape'] = np.array(resized_shape)

        return image, point_cloud