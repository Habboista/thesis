import torch
from torch import Tensor

from ._transform import Transform
from .utils import copy_camera_parameters

class ScaleThroughDepth(Transform):
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def _transform(self, image: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        camera_parameters = copy_camera_parameters(camera_parameters)
        
        
        # Scaling by s is due to point cloud
        scaling_matrix = torch.diag(torch.tensor([1., 1., 1./self.scale_factor]))
        camera_parameters['[R | t]'] = scaling_matrix @ camera_parameters['[R | t]']

        # The downsampling by half is due to the camera focal length
        point_cloud.camera_info['K'] = torch.diag(torch.tensor([0.5, 0.5, 1.])) @ point_cloud.camera_info['K']

        resized_shape: tuple[int, int] = (int(image.shape[-2] * s / 2), int(image.shape[-1] * s / 2))
        image = F.resize(image, resized_shape)

        # Adjust image size
        point_cloud.camera_info['K'][0, 2] *= s
        point_cloud.camera_info['K'][1, 2] *= s

        point_cloud.camera_info['im_shape'] = np.array(resized_shape)
        return image