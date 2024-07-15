import torch
from torch import Tensor
import torch.nn as nn

from .eigen import CoarseFine

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(32, 32))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.base_model(x)
        return self.base_model(x)