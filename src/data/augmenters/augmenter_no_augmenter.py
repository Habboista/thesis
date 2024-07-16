import random

import numpy as np
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._augmenter_abstract import Augmenter
from ..point_cloud import PointCloud

class NoAugmenter(Augmenter):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        return image, point_cloud