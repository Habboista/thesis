from abc import ABC, abstractmethod
from torch import Tensor

from ..point_cloud import PointCloud

class Augmenter(ABC):
    def __init__(self):
        ...
    @abstractmethod
    def __call__(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        ...