import torch
from torch import Tensor
import torch.nn as nn

from ._model_abstract import Model
from .model_eigen import CoarseFine

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(32, 32))

    def _forward(self, x: Tensor, camera_parameters: dict[str, Tensor]) -> Tensor:
        if self.training:
            return self.base_model(x, camera_parameters)
        raise NotImplementedError

        # From coarser scale to finer
            # Detect points of interest

            # Warp

            # Predict

            # Warp back
            
        return self.base_model(x, camera_parameters)