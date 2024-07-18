import torch
from torch import Tensor

from timethis import timethis

__all__ = [
    "render_depth_map",
]

@timethis
def render_depth_map(
    point_cloud: Tensor, camera_parameters: dict[str, Tensor]
) -> Tensor:
    """Project points to the image plane and render their depth map.
    """

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
    depth: Tensor = torch.zeros(tuple(size))
    depth[y, x] = z

    # find the duplicate points and choose the closest depth
    # flat_inds = np.ravel_multi_index((y, x), tuple(size))
    # duplicate_inds, counts = np.unique(flat_inds, return_counts=True)
    # duplicate_inds = duplicate_inds[counts > 1]
    # for dd in duplicate_inds:
    #     pts = np.nonzero(flat_inds == dd)[0]
    #     depth[y[pts], x[pts]] = z[pts].min()

    return depth.unsqueeze(0)