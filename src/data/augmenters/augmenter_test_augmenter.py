import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter
from ..point_cloud import PointCloud

class TestAugmenter(Augmenter):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        # only downsampling by half
        # The downsampling by half is due to the camera focal length
        point_cloud.camera_info['K'] = np.diag([0.5, 0.5, 1.]) @ point_cloud.camera_info['K']

        resized_shape: tuple[int, int] = (int(image.shape[-2] / 2), int(image.shape[-1] / 2))
        image = F.resize(image, resized_shape)

        point_cloud.camera_info['im_shape'] = np.array(resized_shape)
        
        return image, point_cloud