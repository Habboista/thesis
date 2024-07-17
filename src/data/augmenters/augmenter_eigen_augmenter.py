import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter

from ..transforms import horizontal_flip_through_camera
from ..transforms import scale_through_camera
from ..transforms import scale_through_depth

class EigenAugmenter(Augmenter):
    """Jitters color channels, although not the same of the Eigen implementation"""
    def __init__(self):
        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

    def __call__(
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

        return image, point_cloud, camera_parameters