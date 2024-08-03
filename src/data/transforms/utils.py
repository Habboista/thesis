import numpy as np
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

def get_rotation_matrix(x: float, y: float, camera_parameters: dict[str, Tensor]) -> Tensor:
    """Compute the rotation that the camera has to do for making the pixel at (x, y) the principal point."""
    # Compute angles of rotation
    f_x = abs(camera_parameters['K'][0, 0])
    f_y = abs(camera_parameters['K'][1, 1])
    py = int(camera_parameters['K'][1, 2])
    px = int(camera_parameters['K'][0, 2])
    theta_yz = np.arctan2(y - py, f_y)
    theta_xz = np.arctan2(x - px, f_x)
    
    # Rotation matrices
    R_yz = torch.tensor([
        [1,                    0,                   0, 0],
        [0,     np.cos(theta_yz),    np.sin(theta_yz), 0],
        [0,    -np.sin(theta_yz),    np.cos(theta_yz), 0],
        [0,                    0,                   0, 1],
    ], dtype=torch.float32)
    R_xz = torch.tensor([
        [ np.cos(theta_xz), 0,    np.sin(theta_xz), 0],
        [                0, 1,                   0, 0],
        [-np.sin(theta_xz), 0,    np.cos(theta_xz), 0],
        [                0, 0,                   0, 1],
    ], dtype=torch.float32)
    R: Tensor = R_yz @ R_xz

    return R

def get_start_and_end_points(R: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
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
    ])

    # Camera matrix (3x4)
    P = torch.hstack((camera_parameters['K'], torch.zeros(3, 1)))

    # Points projected in warped image
    end_points = corners @ P.T
    end_points[:, :2] /= end_points[:, 2:]
    end_points = end_points[:, :2]

    # Points projected in original image
    start_points = corners @ R.T @ P.T
    start_points[:, :2] /= start_points[:, 2:]
    start_points = start_points[:, :2]

    return start_points, end_points