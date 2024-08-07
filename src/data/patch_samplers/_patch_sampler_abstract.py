from abc import ABC, abstractmethod
from torch import Tensor

from timethis import timethis

class PatchSampler(ABC):
    def __init__(self):
        ...
    
    def _check_inputs(self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]):
        if type(image) != Tensor:
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
        if type(point_cloud) != Tensor:
            raise TypeError(f"point_cloud must be a PointCloud, got {type(point_cloud)}")
        
        if len(image.shape) != 3:
            raise ValueError(f"image must be a 3D tensor of shape 3 x H x W, got {image.shape}")
        if image.shape[0] != 3:
            raise ValueError(f"image must be a 3D tensor of shape 3 x H x W, got {image.shape}")

    def _check_outputs(self, image: Tensor, depth_map: Tensor, camera_parameters: dict[str, Tensor]):
        if len(image.shape) != 4:
            raise ValueError(f"image must be a 4D tensor of shape B x 3 x H x W, got {image.shape}")
        if image.shape[1] != 3:
            raise ValueError(f"image must be a 4D tensor of shape B x 3 x H x W, got {image.shape}")
        
        if len(depth_map.shape) != 4:
            raise ValueError(f"depth_map must be a 4D tensor of shape B x 1 x H x W, got {depth_map.shape}")
        if depth_map.shape[1] != 1:
            raise ValueError(f"depth_map must be a 4D tensor of shape B x 1 x H x W, got {depth_map.shape}")

    @abstractmethod
    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        ...
        
    def __call__(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._check_inputs(image, point_cloud, camera_parameters)
        result = self._call(image, point_cloud, camera_parameters)
        self._check_outputs(*result)
        return result