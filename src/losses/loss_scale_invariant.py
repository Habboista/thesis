import torch
import torch.nn as nn
from torch import Tensor

from ._loss_abstract import Loss

class ScaleInvariantLoss(Loss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
        """Expected pred_depth in log scale and gt_depth in linear scale."""
        mask = (gt_depth > 0.)
        d = pred_depth[mask] - torch.log(gt_depth[mask])
        return torch.mean(d**2) - 0.5 * torch.mean(d)**2