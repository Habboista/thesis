import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from .utils import get_metrics
from ..info import Info
from ..kitti_data.kitti_raw import KITTIRAWDataset
from timethis import timethis

def eval(
        model: nn.Module,
        dataset: KITTIRAWDataset,
        val_info: Info,
) -> None:
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        for image, depth_map in tqdm(dataset):
            # convert to cuda
            image = image.to('cuda')
            depth_map = depth_map.to('cuda')

            # inputs have two leading batch dimensions
            image = image.reshape(-1, *image.shape[2:])
            depth_map = depth_map.reshape(-1, *depth_map.shape[2:])

            # predict
            pred = model(image)
        
            val_info.log_info(get_metrics(depth_map, torch.exp(pred)))