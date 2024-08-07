import torch
from torch import Tensor

from timethis import timethis

@timethis
def get_metrics(pred_depth: Tensor, gt_depth: Tensor) -> dict[str, float]:
    # Mask out of range depth values (following Eigen et al.)
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
    mask = mask & (pred_depth > 0.) # Not present in original evaluation code

    # Evaluate only center crop (following Eigen et al.)
    gt_height, gt_width = gt_depth.shape[-2:]
    crop = [int(0.40810811 * gt_height), int(0.99189189 * gt_height),
            int(0.03594771 * gt_width),  int(0.96405229 * gt_width)]
    crop_mask = torch.zeros_like(mask, dtype=torch.bool)
    crop_mask[..., crop[0]:crop[1], crop[2]:crop[3]] = True
    mask = mask & crop_mask

    # Flatten selected values
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    # Align depth predictions with ground truth
    # TODO

    return compute_metrics(pred_depth, gt_depth)

def compute_metrics(pred: Tensor, gt: Tensor) -> dict[str, float]:
    """Computation of error metrics between predicted and ground truth depths
    """
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred must have the same shape, got {gt.shape} and {pred.shape}")
    
    default_accuracy = 0.
    default_error = float('inf')
    if pred.numel() == 0:
        return {
            'a1': default_accuracy,
            'a2': default_accuracy,
            'a3': default_accuracy,
            'abs_rel': default_error,
            'sq_rel': default_error,
            'rmse': default_error,
            'rmse_log': default_error,
            'si_err': default_error,
        }

    thresh = torch.maximum((gt / pred), (pred / gt))
    mask_a1 = (thresh < 1.25)
    mask_a2 = (thresh < 1.25 ** 2)
    mask_a3 = (thresh < 1.25 ** 3)
    a1 = (mask_a1.mean(dtype=torch.float).cpu().item() if mask_a1.any() else default_accuracy)
    a2 = (mask_a2.mean(dtype=torch.float).cpu().item() if mask_a2.any() else default_accuracy)
    a3 = (mask_a3.mean(dtype=torch.float).cpu().item() if mask_a3.any() else default_accuracy)

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean()).cpu().item()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean()).cpu().item()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt).cpu().item()

    sq_rel = torch.mean(((gt - pred) ** 2) / gt).cpu().item()

    d = torch.log(gt) - torch.log(pred)
    si_err = (torch.mean(d**2) - torch.mean(d)**2).cpu().item()
    return {
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'si_err': si_err,
    }