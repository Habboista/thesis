import torch
import torch.nn as nn
from torch import Tensor

class ScaleInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Expects x and y tensors of the same shape representing
        log depth.
        """
        d = x - y
        return torch.mean(d**2) - 0.5 * torch.mean(d)**2