from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from ..data.patch_samplers.utils import get_blur_weight_mask

class Loss(Module, ABC):

    def __init__(self):
        super().__init__()

    def _check_inputs(self, pred_depth: Tensor, gt_depth: Tensor):
        assert pred_depth.shape == gt_depth.shape, \
            "Expected pred depth map and ground truth depth map of the same shape, " \
                f"got {pred_depth.shape} and {gt_depth.shape}"
        assert len(pred_depth.shape) >= 3, \
            "Expected depth map of size ... x 1 x H x W, " \
                f"got {pred_depth.shape}"
        assert pred_depth.min() > 0., \
            "Expected pred depth map to be strictly positive, " \
                f"but its minimum value is {pred_depth.min()}"

    def _check_output(self, loss: Tensor):
        assert len(loss.shape) == 0, \
            f"Expected loss of shape (,) but got {loss.shape}"
        
    def forward(self, pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
        """Expects pred_depth and gt_depth tensors of same shape ... x 1 x H x W.
        pred_depth and gt_depth must be in linear scale.
        """
        self._check_inputs(pred_depth, gt_depth)
        loss = self._forward(pred_depth, gt_depth)
        self._check_output(loss)
        return loss
    
    @abstractmethod
    def _forward(self, pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
        ...