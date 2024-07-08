from abc import ABC, abstractmethod
from torch import Tensor

from . import PointCloud

class Augmenter(ABC):
    def __init__(self):
        ...
    @abstractmethod
    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, PointCloud]:
        ...