from abc import ABC, abstractmethod
from torch import Tensor

class Augmenter(ABC):
    def __init__(self):
        ...
    @abstractmethod
    def __call__(self, image: Tensor, point_cloud: Tensor) -> tuple[Tensor, Tensor]:
        ...