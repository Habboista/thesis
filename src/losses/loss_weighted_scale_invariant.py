import torch
import torch.nn as nn
from torch import Tensor

from ._loss_abstract import Loss
from ..data.patch_samplers.utils import get_blur_weight_mask

class WeightedScaleInvariantLoss(Loss):
    """Like scale invariant but applies a weighting to the loss
    that is based on the amount of blur applied."""
    def __init__(self):
        super().__init__()

    def _forward(self, pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
        assert pred_depth.min() > 0., \
            "Expected pred depth map to be strictly positive, " \
                f"but its minimum value is {pred_depth.min()}"
        
        mask = (gt_depth > 0.)
        d = torch.log(pred_depth[mask]) - torch.log(gt_depth[mask])
        weights = get_blur_weight_mask(pred_depth)[mask]
        wsum = torch.sum(weights)
        return torch.sum(weights * d**2) / wsum - 0.5 * (torch.sum(weights * d) / wsum)**2