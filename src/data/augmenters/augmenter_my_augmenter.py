import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter
from ..transforms import horizontal_flip_through_camera
from ..transforms import scale_through_camera
from ..transforms import scale_through_depth

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

    def __call__(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        
        # Color jittering
        if self.color_jittering:
            image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if self.hflip and flip:
            image, camera_parameters = horizontal_flip_through_camera(image, camera_parameters)
        
        # Scaling by s
        s = random.uniform(*self.scaling_range)
        image, camera_parameters = scale_through_depth(image, camera_parameters, s)

        return image, point_cloud, camera_parameters