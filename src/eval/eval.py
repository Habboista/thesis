import torch
from torch import Tensor
from tqdm import tqdm

from .utils import get_metrics

from ..models import Model
from ..data.kitti_raw import KITTIRAWDataset
from ..data.patch_samplers import CameraPreservingSampler
from ..info import Info
from ..inference import infer

def eval(
        model: Model,
        dataset: KITTIRAWDataset,
        val_info: Info,
) -> None:
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        for image, depth_map, camera_parameters in tqdm(dataset):
            # image 1 x 3 x H x W
            # depth_map 1 x 1 x H x W

            # convert to cuda
            image = image.to('cuda')
            depth_map = depth_map.to('cuda')

            # predict
            pred = model(image, camera_parameters)
        
            val_info.log_info(get_metrics(pred, depth_map))

def eval_pipeline(
        model: Model,
        dataset: KITTIRAWDataset,
        patch_sampler: CameraPreservingSampler,
        max_num_patches: int,
        num_epochs: int,
        lr: float,
        eval_info: Info,
) -> None:
    model.to('cuda')
    model.eval()

    eval_info.new_epoch()
    for image, depth_map, camera_parameters in tqdm(dataset):
        # image 1 x 3 x H x W
        # depth_map 1 x 1 x H x W

        # convert to cuda
        #image = image.to('cuda')
        #depth_map = depth_map.to('cuda')
        #for k in camera_parameters.keys():
        #    camera_parameters[k] = camera_parameters[k].to('cuda')
        
        # predict
        pred: Tensor = infer(image, camera_parameters, model, patch_sampler, max_num_patches, num_epochs, lr)
    
        eval_info.log_info(get_metrics(pred, depth_map))
    eval_info.print_last_epoch_summary()