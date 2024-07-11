import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional as F
from tqdm import tqdm

from ..models.abstract_model import AbstractModel
from timethis import timethis

@timethis
def compute_depth_errors(gt_depth: Tensor, pred_depth: Tensor) -> dict[str, float]:
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)

    gt_height, gt_width = gt_depth.shape[-2:]
    crop = [int(0.40810811 * gt_height), int(0.99189189 * gt_height),
            int(0.03594771 * gt_width),  int(0.96405229 * gt_width)]
    crop_mask = torch.zeros_like(mask)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = mask & crop_mask

    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    return compute_errors(gt_depth, pred_depth)

def compute_errors(gt: Tensor, pred: Tensor) -> dict[str, float]:
    """Computation of error metrics between predicted and ground truth depths
    """
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
    a2 = (mask_a2.mean().item() if mask_a2.any() else default_accuracy)
    a3 = (mask_a3.mean().item() if mask_a3.any() else default_accuracy)
    a1 = (mask_a1.mean().item() if mask_a1.any() else default_accuracy)

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean()).item()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean()).item()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()

    sq_rel = torch.mean(((gt - pred) ** 2) / gt).item()

    d = torch.log(gt) - torch.log(pred)
    si_err = torch.mean(d**2) - torch.mean(d) ** 2
    return {'a1': a1, 'a2': a2, 'a3': a3, 'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'si_err': si_err}

def eval(
        model: AbstractModel,
        loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    model.to('cuda')
    model.eval()
    print("Validating...")
    with torch.no_grad():
        errors = []
        for image, depth_map, valid_mask, overlap_mask in tqdm(loader):
            # convert to cuda
            image = image.to('cuda')
            depth_map = depth_map.to('cuda')

            # inputs have two leading batching dimensions
            image = image.reshape(-1, *image.shape[2:])
            depth_map = depth_map.reshape(-1, *depth_map.shape[2:])
            valid_mask = valid_mask.reshape(-1, *valid_mask.shape[2:])

            # predict
            pred = model(image)
            pred = F.resize(pred, depth_map.shape[-2:], interpolation=F.InterpolationMode.BILINEAR)
        
            errors.append(compute_depth_errors(depth_map, torch.exp(pred)))
            
    result: dict[str, float] = {k: torch.tensor([e[k] for e in errors]).mean().item() for k in errors[0].keys()}
    return result