import torch
from torch import Tensor

def copy_camera_parameters(camera_parameters: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.clone() for k, v in camera_parameters.items()}

def batched_back_project(camera_parameters: dict[str, Tensor], p: Tensor, Z: Tensor) -> Tensor:
    """
    Args:
        camera_parameters:
            dict of Tensors describing the camera
        p:
            Tensor of shape N x 3, each row is an homogeneous 2-vector
            corresponding to an image location
        Z:
            Tensor of shape N or (,), each value represents the desired Z-coordinate of
            the respective backprojected point from p array in inhomogeneous representation
    
    Returns:
        ...
    
    Raises:
        ...
    """
    if len(p.shape) != 2 or p.shape[1] != 3:
        raise ValueError("p expected to be of shape N x 3")
    
    P = torch.hstack((camera_parameters['K'], torch.zeros((3, 1))))
    P_inv = torch.linalg.inv(P)
    p = p @ P_inv.T # N x 4
    C = torch.tensor([0., 0., 0., 1.])

    _lambda = (Z * p[..., 3] - p[..., 2]) / (C[None, 2] - Z * C[None, 3]) # N values

    p = p + _lambda[..., None] * C[None, :] # (N x 4) + (N x 1) * (1 x 4)
    p = p[:, :3] / p[:, 3:] 
    
    return p