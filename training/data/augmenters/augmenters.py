from .. import Augmenter
from torch import Tensor

class TrivialAugmenter(Augmenter):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, point_cloud: Tensor) -> tuple[Tensor, Tensor]:
        return image, point_cloud