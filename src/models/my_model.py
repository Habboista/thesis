import torch
from torch import Tensor
import torch.nn as nn

from .eigen import CoarseFine

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(150, 150))
    
    def to(self, device):
        """It does nothing. Always working with CUDA."""
        pass

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.base_model(x)
        raise NotImplementedError