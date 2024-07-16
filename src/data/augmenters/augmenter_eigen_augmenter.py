import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter
from ..point_cloud import PointCloud

class EigenAugmenter(Augmenter):
    """Jitters color channels, although not the same of the Eigen implementation"""
    def __init__(self):
        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        # Color jittering
        image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if flip:
            point_cloud.camera_info['K'][0, 0] *= -1
            point_cloud.camera_info['K'][0, 2] = point_cloud.camera_info['im_shape'][1] - point_cloud.camera_info['K'][0, 2]
            image = F.hflip(image)
        
        # Scaling by s and downsampling by half
        # the reason is that if all the scaling is due to the point cloud
        # then the depth range is too large
        # the focal length of the matrix is consistent across samples
        
        # Scaling by s is due to point cloud
        s = random.uniform(1., 1.2)
        point_cloud.points[:, 0] /= s

        # The downsampling by half is due to the camera focal length
        point_cloud.camera_info['K'] = np.diag([0.5, 0.5, 1.]) @ point_cloud.camera_info['K']

        resized_shape: tuple[int, int] = (int(image.shape[-2] * s / 2), int(image.shape[-1] * s / 2))
        image = F.resize(image, resized_shape)

        # Adjust image size
        point_cloud.camera_info['K'][0, 2] *= s
        point_cloud.camera_info['K'][1, 2] *= s

        point_cloud.camera_info['im_shape'] = np.array(resized_shape)

        return image, point_cloud