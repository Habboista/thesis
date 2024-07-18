import torch
import torch.nn as nn
from torch import Tensor

from ._loss_abstract import Loss

class ScaleInvariantLoss(Loss):
    def __init__(self):
        super().__init__()

    def _forward(self, pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
        #assert pred_depth.min() > 0., \
        #    "Expected pred depth map to be strictly positive, " \
        #        f"but its minimum value is {pred_depth.min()}"
        
        mask = (gt_depth > 0.)
        d = pred_depth[mask] - gt_depth[mask]
        return torch.mean(d**2) - 0.5 * torch.mean(d)**2