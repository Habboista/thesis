import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .abstract_augmenter import Augmenter
from ..point_cloud import PointCloud

class NoAugmenter(Augmenter):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        return image, point_cloud
    
class EigenAugmenter(Augmenter):
    """Jitters color channels, although not the same of the Eigen implementation"""
    def __init__(self):
        self.color_jitter = T.ColorJitter(0.1, 0.1, 0.1, 0.)

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        # Color jittering
        image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if flip:
            point_cloud.points[:, 1] *= -1
            image = F.hflip(image)
        
        # Scaling
        s = random.uniform(0.5, 0.75)
        resized_shape: tuple[int, int] = (int(image.shape[-2] * s), int(image.shape[-1] * s))
        image = F.resize(image, resized_shape)

        point_cloud.points[:, 0] = (point_cloud.points[:, 0] - 0.27) / s + 0.27
        point_cloud.projection_matrix = np.diag([s, s, 1.]) @ point_cloud.projection_matrix

        return image, point_cloud