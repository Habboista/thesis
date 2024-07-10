import torch
import torch.nn as nn

from ..kitti_data.kitti_raw.kitti_dataset import KITTIRAWDataset

def eval(
        model: nn.Module,
        dataset: KITTIRAWDataset,
):
    raise NotImplementedError
        