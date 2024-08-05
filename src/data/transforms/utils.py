import numpy as np
import torch
from torch import Tensor

__all__ = [
    "copy_camera_parameters",
    "batched_back_project",
    "get_rotation_matrix",
]

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
        The homogeneous points in the absolute reference system
    
    Raises:
        ValueError if p is not of shape N x 3, i.e. expressed in homogeneous coordinates
    """
    if len(p.shape) != 2 or p.shape[1] != 3:
        raise ValueError("p expected to be of shape N x 3")
    
    K = camera_parameters['K']
    device: torch.device = K.device
    assert p.device == device and Z.device == device

    P = torch.hstack((K, torch.zeros(3, 1, device=device)))
    P_inv = torch.linalg.pinv(P)
    p = p @ P_inv.T # N x 4
    C = torch.tensor([0., 0., 0., 1.], device=device)

    _lambda = (Z * p[..., 3] - p[..., 2]) / (C[None, 2] - Z * C[None, 3]) # N values

    p = p + _lambda[..., None] * C[None, :] # (N x 4) + (N x 1) * (1 x 4)
    p = p[:, :] / p[:, 3:]
    
    p = p @ torch.linalg.inv(camera_parameters['[R | t]']).T

    return p

def get_rotation_matrix(x: float, y: float, camera_parameters: dict[str, Tensor]) -> Tensor:
    """Compute the rotation that the camera has to do for making the pixel at (x, y) the principal point.
    
    Args:
        x: x-coordinate of the pixel (column)
        y: y-coordinate of the pixel (row)
        camera_parameters:
            dict of Tensors describing the camera
    
    Returns:
        The 3x3 rotation matrix as a torch tensor on the same device of the camera parameters
    """
    # Compute angles of rotation
    K = camera_parameters['K']
    device = K.device
    f_x: float = abs(K[0, 0].item())
    f_y: float = abs(K[1, 1].item())
    py: float = K[1, 2].item()
    px: float = K[0, 2].item()
    theta_yz: float = np.arctan2(y - py, f_y)
    theta_xz: float = np.arctan2(x - px, f_x)
    
    # Rotation matrices
    R_yz = torch.tensor([
        [1,                    0,                   0, 0],
        [0,     np.cos(theta_yz),    np.sin(theta_yz), 0],
        [0,    -np.sin(theta_yz),    np.cos(theta_yz), 0],
        [0,                    0,                   0, 1],
    ], dtype=torch.float32, device=device)
    R_xz = torch.tensor([
        [ np.cos(theta_xz), 0,    np.sin(theta_xz), 0],
        [                0, 1,                   0, 0],
        [-np.sin(theta_xz), 0,    np.cos(theta_xz), 0],
        [                0, 0,                   0, 1],
    ], dtype=torch.float32, device=device)
    
    R: Tensor = R_yz @ R_xz
    return R

def get_start_and_end_points(R: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    # Camera matrix (3x4)
    K = camera_parameters['K']
    assert K.device == R.device, f"R and K must be on the same device, got {K.device} and {R.device}"
    device = K.device

    P = torch.hstack((K, torch.zeros(3, 1, device=device)))

    # Arbitrary points in space for computing the homography
    # They just need to be a homogeneous reference system
    # i.e. each three of them is a group of independent vectors
    # (x, y, z, 1)
    offset = 1.
    corners = torch.tensor([
        [-offset, -offset, offset*10., 1.],
        [-offset,  offset, offset*10., 1.],
        [ offset, -offset, offset*10., 1.],
        [ offset,  offset, offset*10., 1.],
    ], device=device)

    # Points projected in warped image
    end_points = corners @ P.T
    end_points[:, :2] /= end_points[:, 2:]
    end_points = end_points[:, :2]

    # Points projected in original image
    start_points = corners @ R.T @ P.T
    start_points[:, :2] /= start_points[:, 2:]
    start_points = start_points[:, :2]

    return start_points, end_points