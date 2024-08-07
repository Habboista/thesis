import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter
from ..transforms import scale_through_camera

class TestAugmenter(Augmenter):
    def __init__(self):
        pass

    
    def __call__(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        
        return image, point_cloud, camera_parameters