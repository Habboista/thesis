import random

import numpy as np
import skimage
import torch
from torch import Tensor
import torchvision.transforms.functional as F

from .abstract_patch_sampler import PatchSampler
from ..point_cloud import PointCloud
from .utils import blur

class MyPatchSampler(PatchSampler):
    def __init__(
            self,
            batch_size: int,
            blur: bool,
            half_h: int,
            half_w: int,
            corner_sampling: bool,
        ):
        self.half_h = half_h
        self.half_w = half_w
        self.batch_size = batch_size
        self.blur = blur
        self.corner_sampling = corner_sampling

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:

        # Sample points of interest and warp the image and depth centering them
        if not self.corner_sampling:
            samples = self.random_sampling(image, point_cloud)
        else:
            samples = self.sample_corner(image, point_cloud)

        # Crop around the central point of the image (as defined by the camera)
        image_crops = [s[0]\
            [...,
            int(point_cloud.camera_info['K'][1, 2] - self.half_h):int(point_cloud.camera_info['K'][1, 2] + self.half_h),
            int(point_cloud.camera_info['K'][0, 2] - self.half_w):int(point_cloud.camera_info['K'][0, 2] + self.half_w)]
            for s in samples]
        depth_map_crops = [s[1]\
            [...,
            int(point_cloud.camera_info['K'][1, 2] - self.half_h):int(point_cloud.camera_info['K'][1, 2] + self.half_h),
            int(point_cloud.camera_info['K'][0, 2] - self.half_w):int(point_cloud.camera_info['K'][0, 2] + self.half_w)]
            for s in samples]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return blur(image) if self.blur else image, depth_map

    def random_sampling(self, image: Tensor, point_cloud: PointCloud) -> list[tuple[Tensor, Tensor]]:
        # Randomly select point
        p1 = 0.2 # TODO: p1, p2 should depend on image size
        p2 = 0.8
        def sample_point():
            x = random.randrange(int(p1 * image.shape[-1]), int(p2 * image.shape[-1]))
            y = random.randrange(int(p1 * image.shape[-2]), int(p2 * image.shape[-2]))
            return x, y

        return [self.warp(image, point_cloud, *sample_point()) for _ in range(self.batch_size)]
    
    def sample_corner(self, image: Tensor, point_cloud: PointCloud) -> list[tuple[Tensor, Tensor]]:
        assert len(image.shape) == 3
        assert image.shape[0] == 3

        np_image = image.permute(1, 2, 0).cpu().numpy()
        np_corner_response = skimage.feature.corner_moravec(skimage.color.rgb2gray(np_image))

        # Cancel response near image boundaries (otherwise the warp contains out-of-view points)
        # TODO: use a different shape for the allowed area than the rectangle
        p1 = 0.2
        p2 = 0.8
        i1 = int(p1 * image.shape[-1])
        i2 = int(p2 * image.shape[-1])
        j1 = int(p1 * image.shape[-2])
        j2 = int(p2 * image.shape[-2]) 
        np_corner_response[:i1, :] = 0
        np_corner_response[i2:, :] = 0
        np_corner_response[:, :j1] = 0
        np_corner_response[:, j2:] = 0

        # Sample peaks of interest
        peaks = skimage.feature.corner_peaks(np_corner_response) # TODO: this can be empty
        indeces = random.choices(range(peaks.shape[0]), k=self.batch_size)
    
        return [self.warp(image, point_cloud, x, y) for x, y in peaks[indeces]]
    
    def warp(self, image: Tensor, point_cloud: PointCloud, x: int, y: int) -> tuple[Tensor, Tensor]:
        """Given a point (x, y) applies perspective transform to both the image and the point cloud
        so that the point matches the central point (defined by px, py parameters of the camera) of the image."""
        
        # Compute angles of rotation
        alpha_x = abs(point_cloud.camera_info['K'][0, 0])
        alpha_y = abs(point_cloud.camera_info['K'][1, 1])
        py = int(point_cloud.camera_info['K'][1, 2])
        px = int(point_cloud.camera_info['K'][0, 2])
        theta_yz = -np.arctan2(y - py, alpha_y)
        theta_xz = -np.arctan2(x - px, alpha_x)
        
        # Rotation matrices
        R_yz = np.array([
            [1,                 0,                0, 0],
            [0,  np.cos(theta_yz), np.sin(theta_yz), 0],
            [0, -np.sin(theta_yz), np.cos(theta_yz), 0],
            [0,                 0,                0, 1],
        ])
        R_xz = np.array([
            [ np.cos(theta_xz), 0, np.sin(theta_xz), 0],
            [                0, 1,                0, 0],
            [-np.sin(theta_xz), 0, np.cos(theta_xz), 0],
            [0,                 0,                0, 1],
        ])

        # Arbitrary points in space for computing the homography
        # They just need to be a homogeneous reference system
        # i.e. each three of them is a group of independent vectors
        offset = 1
        corners = np.array([
            [-offset, -offset, offset*10, 1],
            [-offset,  offset, offset*10, 1],
            [ offset, -offset, offset*10, 1],
            [ offset,  offset, offset*10, 1],
        ])

        # Camera matrix (3x4)
        P = np.hstack((point_cloud.camera_info['K'], np.zeros((3, 1))))

        # Points projected in warped image
        end_points = corners @ P.T
        end_points[:, :2] /= end_points[:, 2:]
        end_points = end_points[:, :2]

        # Points projected in original image
        start_points = corners @ R_xz.T @ R_yz.T @ P.T
        start_points[:, :2] /= start_points[:, 2:]
        start_points = start_points[:, :2]

        # Warp image
        warped = F.perspective(image, start_points, end_points)

        # Compute depth map
        point_cloud.camera_info['R'] = np.linalg.inv(R_xz[:3, :3] @ R_yz[:3, :3])
        depth_map: Tensor = point_cloud.to_depth_map()
        point_cloud.camera_info['R'] = np.eye(3) # Reset for next image
        
        return warped, depth_map