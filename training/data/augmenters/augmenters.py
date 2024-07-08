import random

from torch import Tensor
import torchvision.transforms.functional as F

from .. import Augmenter
from .. import PointCloud

class EigenAugmenter(Augmenter):
    def __init__(self):
        self.crop_size = (172, 576)

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        point_cloud = point_cloud.copy()

        s = random.uniform(1., 1.5)
        shape = image.shape[-2:]
        image = F.resize(image, (shape[0] * s, shape[1] * s))
        point_cloud.points[:, 2] /= s

        return image, point_cloud