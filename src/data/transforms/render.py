import torch
from torch import Tensor

from .utils import batched_back_project
from timethis import timethis

__all__ = [
    "cloud2depth",
    "depth2cloud",
]

@timethis
def cloud2depth(
    point_cloud: Tensor, camera_parameters: dict[str, Tensor]
) -> Tensor:
    """Project points to the image plane and render their depth map.
    """
    # Work only on cpu
    assert point_cloud.device == torch.device('cpu')
    for k in camera_parameters:
        assert camera_parameters[k].device == torch.device('cpu')

    # project the points to camera reference system
    camera_point_cloud = point_cloud @ camera_parameters["[R | t]"].T
    camera_point_cloud = camera_point_cloud[:, :3] / camera_point_cloud[:, 3:]
    
    # project points to image
    img_points = (camera_point_cloud @ camera_parameters['K'].T)
    img_points[:, :2] = img_points[:, :2] / img_points[:, 2:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    size = camera_parameters['image_size']
    x = (torch.round(img_points[:, 0]) - 1).long()
    y = (torch.round(img_points[:, 1]) - 1).long()
    z = camera_point_cloud[:, 2]
    valid_inds = (x >= 0) & (y >= 0) & (x < size[1]) & (y < size[0])
    
    x = x[valid_inds]
    y = y[valid_inds]
    z = z[valid_inds]

    # draw depth map
    depth: Tensor = torch.zeros(*tuple(size))
    depth[y, x] = z

    # TODO: make the following faster and in torch
    # find the duplicate points and choose the closest depth
    # flat_inds = np.ravel_multi_index((y, x), tuple(size))
    # duplicate_inds, counts = np.unique(flat_inds, return_counts=True)
    # duplicate_inds = duplicate_inds[counts > 1]
    # for dd in duplicate_inds:
    #     pts = np.nonzero(flat_inds == dd)[0]
    #     depth[y[pts], x[pts]] = z[pts].min()

    return depth.unsqueeze(0)

def depth2cloud(depth_map: Tensor, camera_parameters: dict[str, Tensor]) -> Tensor:
    """Project depth map to 3D point cloud.
    
    Args:
        depth_map:
            Tensor of shape 1 x H x W
        camera_parameters:
            dict of Tensors describing the camera
        
    Returns:
        Tensor of shape N x 3
    """
    # Work only on cpu
    assert depth_map.device == torch.device('cpu')
    for k in camera_parameters:
        assert camera_parameters[k].device == torch.device('cpu')

    depth_map = depth_map.squeeze(0)

    x: Tensor = torch.linspace(0., float(camera_parameters['image_size'][1] - 1), int(2 * camera_parameters['image_size'][1].item()))
    y: Tensor = torch.linspace(0., float(camera_parameters['image_size'][0] - 1), int(2 * camera_parameters['image_size'][0].item()))

    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    x = grid_x.flatten()
    y = grid_y.flatten()
    
    # TODO use bilinear interpolation, right now is NN interpolation
    y = torch.clamp(y.round(), 0, depth_map.shape[0] - 1).long()
    x = torch.clamp(x.round(), 0, depth_map.shape[1] - 1).long()
    z: Tensor = depth_map[y, x].cpu()
    p = torch.stack((x, y, torch.ones(len(x))), dim=1) # homogeneous

    return batched_back_project(camera_parameters, p, z)