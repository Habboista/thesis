# TODO: remove this module

import numpy as np
import torch
from torch import Tensor
from timethis import timethis

class PointCloud:
    def __init__(
            self,
            points: np.ndarray,
            camera_info: dict[str, np.ndarray],
        ):
        # Decompose camera matrix
        K = camera_info['P_rect'][:, :3]
        R = np.eye(3)
        p4 = camera_info['P_rect'][:, 3]

        t = np.array([
            [p4[0] / K[0, 0]],
            [0.             ],
            [0.             ],
        ])

        camera_info['K'] = K
        camera_info['R'] = R
        camera_info['t'] = t

        del camera_info['P_rect']

        self.camera_info: dict[str, np.ndarray] = camera_info
        self.points: np.ndarray = points

    @timethis
    def to_depth_map(self) -> Tensor:
        K = self.camera_info['K']
        R = self.camera_info['R']
        t = self.camera_info['t']
        
        R_T_velo2cam = self.camera_info['R_T_velo2cam']
        R_cam2rect = self.camera_info['R_cam2rect']
        R_T_rect2rect = np.vstack((
            np.hstack((R,        t)),
            np.array([0, 0, 0, 1.0])
        ))

        velo2rect = R_T_rect2rect @ R_cam2rect @ R_T_velo2cam
        projection_matrix = np.hstack((K, np.zeros((3, 1))))

        # project the points to camera reference system
        rect_points = (velo2rect @ self.points.T).T
        rect_points[:, :3] = rect_points[:, :3] / rect_points[:, 3][..., np.newaxis]
        
        # project points to image
        img_points = (rect_points @ projection_matrix.T)
        img_points[:, :2] = img_points[:, :2] / img_points[:, 2][..., np.newaxis]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        shape = self.camera_info['im_shape']
        x = (np.round(img_points[:, 0]) - 1).astype(np.int_)
        y = (np.round(img_points[:, 1]) - 1).astype(np.int_)
        z = rect_points[:, 2]
        valid_inds = (x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0])
        
        x = x[valid_inds]
        y = y[valid_inds]
        z = z[valid_inds]

        # draw depth map
        depth: np.ndarray = np.zeros(shape)
        depth[y, x] = z

        # find the duplicate points and choose the closest depth
        flat_inds = np.ravel_multi_index((y, x), tuple(shape))
        duplicate_inds, counts = np.unique(flat_inds, return_counts=True)
        duplicate_inds = duplicate_inds[counts > 1]
        for dd in duplicate_inds:
            pts = np.nonzero(flat_inds == dd)[0]
            depth[y[pts], x[pts]] = z[pts].min()

        return torch.from_numpy(depth).unsqueeze(0)