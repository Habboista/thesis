import torch
from tqdm import tqdm

from .utils import get_metrics

from ..models import Model
from ..data.kitti_raw import KITTIRAWDataset
from ..info import Info

def eval(
        model: Model,
        dataset: KITTIRAWDataset,
        val_info: Info,
) -> None:
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        for image, depth_map, camera_parameters in tqdm(dataset):
            # convert to cuda
            image = image.to('cuda')
            depth_map = depth_map.to('cuda')

            # predict
            pred = model(image, camera_parameters)
        
            val_info.log_info(get_metrics(pred, depth_map))