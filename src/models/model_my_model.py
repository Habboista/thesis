import torch
from torch import Tensor
import torch.nn as nn

from ._model_abstract import Model
from .model_eigen import CoarseFine
from ..data.transforms import scale_through_depth

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(32, 32))

    def _forward(self, x: Tensor, camera_parameters: dict[str, Tensor]) -> Tensor:
        if self.training:
            return self.base_model(x, camera_parameters)
        raise NotImplementedError

        # From coarser scale to finer
        scaled_images = [scale_through_depth(x, camera_parameters, s) for s in [0.25, 0.5, 1.]]
        for image, camera_parameters in scaled_images:
            # Detect points of interest
            i, j = (10, 10)

            # Warp
            

            # Blur

            # Predict

            # Warp back

        return self.base_model(x, camera_parameters)