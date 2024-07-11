import numpy as np
from timethis import timethis

class PointCloud:
    def __init__(self, points: np.ndarray, projection_matrix: np.ndarray):
        self.points: np.ndarray = points
        self.projection_matrix: np.ndarray = projection_matrix
    
    def copy(self) -> 'PointCloud':
        return PointCloud(self.points, self.projection_matrix.copy())
    
    @timethis
    def to_depth_map(self, shape: tuple[int, int]) -> np.ndarray:
        # project the points to the image plane
        img_points = (self.points @ self.projection_matrix.T)
        img_points[:, :2] = img_points[:, :2] / img_points[:, 2][..., np.newaxis]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        x = (np.round(img_points[:, 0]) - 1).astype(np.int_)
        y = (np.round(img_points[:, 1]) - 1).astype(np.int_)
        z = self.points[:, 0]
        valid_inds = (x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0])
        
        x = x[valid_inds]
        y = y[valid_inds]
        z = z[valid_inds]

        # draw depth map
        depth: np.ndarray = np.zeros(shape)
        depth[y, x] = z

        # find the duplicate points and choose the closest depth
        flat_inds = np.ravel_multi_index((y, x), shape)
        duplicate_inds, counts = np.unique(flat_inds, return_counts=True)
        duplicate_inds = duplicate_inds[counts > 1]
        for dd in duplicate_inds:
            pts = np.nonzero(flat_inds == dd)[0]
            depth[y[pts], x[pts]] = z[pts].min()

        return depth