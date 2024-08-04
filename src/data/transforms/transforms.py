import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Callable

from .utils import get_rotation_matrix, get_start_and_end_points
from timethis import timethis

__all__ = [
    "horizontal_flip_through_camera",
    "scale_through_camera",
    "scale_through_depth",
    "center_crop_through_camera",
    "blur",
    "warp",
]

def copy_camera_parameters(camera_parameters: dict[str, Tensor]) -> dict[str, Tensor]:
    """Returns a deep copy of the dict of Tensors."""
    return {k: v.clone() for k, v in camera_parameters.items()}

@timethis
def horizontal_flip_through_camera(
    image: Tensor, camera_parameters: dict[str, Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Horizontally flip the image."""
    
    out_camera_parameters: dict[str, Tensor] = copy_camera_parameters(camera_parameters)

    out_camera_parameters['K'][0, 0] *= -1
    out_camera_parameters['K'][0, 2] = out_camera_parameters['image_size'][1] - out_camera_parameters['K'][0, 2]
    
    out_image: Tensor = F.hflip(image)

    return out_image, out_camera_parameters

@timethis
def scale_through_camera(
    image: Tensor, camera_parameters: dict[str, Tensor], s: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """Scale the image by changing the focal length of the camera."""
    
    out_camera_parameters: dict[str, Tensor] = copy_camera_parameters(camera_parameters)
    
    # Scale camera focal length
    out_camera_parameters['K'][0, 0] *= s # f_x
    out_camera_parameters['K'][1, 1] *= s # f_y

    # Adjust image size
    out_camera_parameters['K'][0, 2] *= s # p_x
    out_camera_parameters['K'][1, 2] *= s # p_y
    out_camera_parameters['image_size'] = (out_camera_parameters['image_size'] * s).long()
    
    # Scale image
    out_image: Tensor = F.resize(image, out_camera_parameters['image_size'])

    return out_image, out_camera_parameters

@timethis
def scale_through_depth(
    image: Tensor, camera_parameters: dict[str, Tensor], s: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """Scale the image by scaling distances."""
    out_camera_parameters: dict[str, Tensor] = copy_camera_parameters(camera_parameters)
    
    # Add scaling transformation to the point cloud
    scaling_matrix = torch.diag(torch.tensor([1., 1., 1./s, 1.]))
    out_camera_parameters['[R | t]'] = scaling_matrix @ out_camera_parameters['[R | t]']

    # Adjust image size
    out_camera_parameters['K'][0, 2] *= s # p_x
    out_camera_parameters['K'][1, 2] *= s # p_y
    out_camera_parameters['image_size'] = (out_camera_parameters['image_size'] * s).long()

    # Scale image
    out_image = F.resize(image, out_camera_parameters['image_size'])

    return out_image, out_camera_parameters

@timethis
def center_crop_through_camera(
    image: Tensor, camera_parameters: dict[str, Tensor], crop_size: tuple[int, int]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Center crop the image around the camera principal point."""

    out_camera_parameters: dict[str, Tensor] = copy_camera_parameters(camera_parameters)

    p_x = out_camera_parameters['K'][0, 2]
    p_y = out_camera_parameters['K'][1, 2]
    i = int(p_y - crop_size[0]//2)
    j = int(p_x - crop_size[1]//2)
    out_image = image[
        ...,
        i : i+crop_size[0],
        j : j+crop_size[1],
    ].clone()

    p_x -= (p_x - crop_size[0] // 2) # p_x
    p_y -= (p_y - crop_size[1] // 2) # p_y

    out_camera_parameters['K'][0, 2] = p_x
    out_camera_parameters['K'][1, 2] = p_y
    
    out_camera_parameters['image_size'] = torch.tensor(crop_size).long()
   
    return out_image, out_camera_parameters

@timethis
def blur(
    image: Tensor,
) -> Tensor:
    """Radially (almost) blur the image around the camera principal point."""

    p = torch.tensor([1/6, 2/6, 2.5/6])

    blur: list[Tensor] = [image]
    for _ in range(len(p)):
        blur.append(F.gaussian_blur(blur[-1], (5, 5), (1.5, 1.5)))

    out_image: Tensor = blur[-1]
    for pp, bb in zip(p, blur[-2:-1:-1]):
        i1 = int(image.shape[-2] * pp)
        j1 = int(image.shape[-1] * pp)
        i2 = int(image.shape[-2] * (1 - pp))
        j2 = int(image.shape[-1] * (1 - pp))
        out_image[..., i1:i2, j1:j2] = bb[..., i1:i2, j1:j2]

    return out_image

@timethis
def warp(
    image: Tensor, camera_parameters: dict[str, Tensor], x: int, y: int, interpolation: T.InterpolationMode,
) -> tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, dict[str, Tensor], Callable, Callable]:
    """Given a point (x, y) applies perspective transform to both the image and the point cloud
    so that the point matches the central point (defined by px, py parameters of the camera) of the image.
    """
    
    out_camera_parameters: dict[str, Tensor] = copy_camera_parameters(camera_parameters)

    # Compute rotation matrix
    R: Tensor = get_rotation_matrix(x, y, camera_parameters)

    # Correspond points in the image and the warped image for computing homography
    start_points, end_points = get_start_and_end_points(R, out_camera_parameters)

    # Warp image
    out_image = F.perspective(image, start_points, end_points, interpolation=interpolation)

    # Adjust camera parameters
    out_camera_parameters['[R | t]'] = \
        torch.linalg.inv(R) @ out_camera_parameters['[R | t]']

    return out_image, out_camera_parameters