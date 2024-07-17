import torch
from torch import Tensor
import torchvision.transforms.functional as F

from ._patch_sampler_abstract import PatchSampler
from ..transforms import render_depth_map

class NoPatchSampler(PatchSampler):
    def __init__(self):
        pass

    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        
        depth_map: Tensor = render_depth_map(point_cloud, camera_parameters)

        camera_parameters_list = [camera_parameters]
        batched_camera_parameters = {
            k: torch.stack([param[k] for param in camera_parameters_list])
            for k in camera_parameters.keys()
        }
        return image.unsqueeze(0), depth_map.unsqueeze(0), camera_parameters