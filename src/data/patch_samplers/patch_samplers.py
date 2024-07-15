import random

import numpy as np
import skimage
import torch
from torch import Tensor
import torchvision.transforms.functional as F

from .abstract_patch_sampler import PatchSampler
from .utils import blur
from ..point_cloud import PointCloud
from timethis import timethis

@timethis
class NoPatchSampler(PatchSampler):
    def __init__(self):
        pass

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        depth_map: Tensor = point_cloud.to_depth_map()

        return image.unsqueeze(0), depth_map.unsqueeze(0)

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
        print("image shape before crop",image.shape[-2:])

        # Sample points of interest and warp the image and depth centering them
        sample = self.random_sampling if not self.corner_sampling else self.sample_corner
        samples = [sample(image, point_cloud) for _ in range(self.batch_size)]

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

    def random_sampling(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        # Randomly select point
        p1 = 0.2 # TODO: p1, p2 should depend on image size
        p2 = 0.8
        x = random.randrange(int(p1 * image.shape[-1]), int(p2 * image.shape[-1]))
        y = random.randrange(int(p1 * image.shape[-2]), int(p2 * image.shape[-2]))

        return self.warp(image, point_cloud, x, y)
    
    def sample_corner(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        assert len(image.shape) == 3
        assert image.shape[0] == 3

        np_image = image.permute(1, 2, 0).cpu().numpy()
        np_corner_response = skimage.feature.corner_moravec(skimage.color.rgb2gray(np_image))

        # Cancel response near image boundaries (otherwise the warp contains out-of-view points)
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
        peaks = skimage.feature.corner_peaks(np_corner_response) # TODO: this can be empty
        print(peaks.shape)
        print(np_corner_response.min(), np_corner_response.max())
        i = random.choice(range(peaks.shape[0]))
        x = peaks[i, 0]
        y = peaks[i, 1]

        return self.warp(image, point_cloud, x, y)
    
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


class TestPatchSampler(PatchSampler):
    def __init__(self):
        self.crop_size = (172, 576)

    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        # project point cloud to image
        depth_map: Tensor = point_cloud.to_depth_map()

        # Generate center crop # TODO crop around central point
        image = F.center_crop(image, self.crop_size)
        depth_map = F.center_crop(depth_map, self.crop_size)

        # Batch output
        image = image.unsqueeze(0)
        depth_map = depth_map.unsqueeze(0)

        return image, depth_map
    
class EigenPatchSampler(PatchSampler):
    """Reproduces Eigen augmentations, except for color jittering which is in the Augmenter class"""
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)

    @timethis
    def _call(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor]:
        # project point cloud to image
        depth_map: Tensor = point_cloud.to_depth_map()

        # Generate crops
        ii = random.choices(range(image.shape[-2] - self.crop_size[0] + 1), k=self.batch_size)
        jj = random.choices(range(image.shape[-1] - self.crop_size[1] + 1), k=self.batch_size)

        image_crops = [F.crop(image, i, j, *self.crop_size) for i, j in zip(ii, jj)]
        depth_map_crops = [F.crop(depth_map, i, j, *self.crop_size) for i, j in zip(ii, jj)]

        # Batch crops
        image = torch.stack(image_crops)
        depth_map = torch.stack(depth_map_crops)

        return image, depth_map