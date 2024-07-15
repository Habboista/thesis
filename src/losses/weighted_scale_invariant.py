import torch
import torch.nn as nn
from torch import Tensor

from ..data.patch_samplers.utils import get_blur_weight_mask

class WeightedScaleInvariantLoss(nn.Module):
    """Like scale invariant but applies a weighting to the loss
    that is based on the amount of blur applied."""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Expects x and y tensors of the same shape representing
        log depth.
        """
        d = x - y
        weights = get_blur_weight_mask(x)
        wsum = torch.sum(weights)
        return torch.sum(weights * d**2) / wsum - 0.5 * (torch.sum(weights * d) / wsum)**2