import random

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
    def __init__(self):
        self.color_jitter = T.ColorJitter(0.1, 0.1, 0.1, 0.)

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        image = self.color_jitter(image)
        return image, point_cloud