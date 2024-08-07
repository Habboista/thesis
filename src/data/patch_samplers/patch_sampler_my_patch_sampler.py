import random

import numpy as np
import skimage
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._patch_sampler_abstract import PatchSampler
from .utils import clean_corner_response
from ..transforms import cloud2depth
from ..transforms import warp
from ..transforms import blur
from ..transforms import center_crop_through_camera
from ..transforms import horizontal_flip_through_camera
from ..transforms import scale_through_depth

class MyPatchSampler(PatchSampler):
    def __init__(
            self,
            color_jittering: bool,
            hflip: bool,
            scaling_range: tuple[float, float],
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

        self.color_jittering = color_jittering
        self.hflip = hflip
        self.scaling_range = scaling_range

        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # Color jittering
        if self.color_jittering:
            image = self.color_jitter(image)

        # Horizontal Flip
        flip = random.random() > 0.5
        if self.hflip and flip:
            image, camera_parameters = horizontal_flip_through_camera(image, camera_parameters)
        
        # Scaling by s
        s = random.uniform(*self.scaling_range)
        image, camera_parameters = scale_through_depth(image, camera_parameters, s)

        # Sample points of interest and warp the image and depth centering them
        if not self.corner_sampling:
            samples = self.random_sampling(image, camera_parameters)
        else:
            samples = self.sample_corner(image, camera_parameters)

        out_images: list[Tensor] = []
        out_camera_parameters: list[dict[str, Tensor]] = []
        out_depth_maps: list[Tensor] = []
        for img, cam_params in samples:
            # Center crop
            out_img, out_cam_params = center_crop_through_camera(img, cam_params, (self.half_h*2+1, self.half_w*2+1))
            #out_img, out_cam_params = img, cam_params

            # Blur
            if self.blur:
                out_img = blur(out_img)
            out_images.append(out_img)
            out_camera_parameters.append(out_cam_params)
            
            # Render depth
            out_depth_maps.append(cloud2depth(point_cloud, out_cam_params))
        
        # Batch crops
        batched_images: Tensor = torch.stack(out_images)
        batched_depth_maps: Tensor = torch.stack(out_depth_maps)
        batched_camera_parameters: dict[str, Tensor] = {
            k: torch.stack([param[k] for param in out_camera_parameters])
            for k in camera_parameters.keys()
        }

        return batched_images, batched_depth_maps, batched_camera_parameters

    def random_sampling(
        self, image: Tensor, camera_parameters: dict[str, Tensor]
    ) -> list[tuple[Tensor, dict[str, Tensor]]]:
        
        # Randomly select point
        p1 = 0.2 # TODO: p1, p2 should depend on image size
        p2 = 0.8
        def sample_point() -> tuple[int, int]:
            x = random.randrange(int(p1 * image.shape[-1]), int(p2 * image.shape[-1]))
            y = random.randrange(int(p1 * image.shape[-2]), int(p2 * image.shape[-2]))
            return x, y

        return [warp(image, camera_parameters, *sample_point()) for _ in range(self.batch_size)]
    
    def sample_corner(
        self, image: Tensor, camera_parameters: dict[str, Tensor]
    ) -> list[tuple[Tensor, dict[str, Tensor]]]:
        assert len(image.shape) == 3
        assert image.shape[0] == 3

        np_image = image.permute(1, 2, 0).cpu().numpy()
        np_corner_response = skimage.feature.corner_moravec(skimage.color.rgb2gray(np_image))

        np_corner_response = clean_corner_response(np_corner_response)

        # Sample peaks of interest
        peaks = skimage.feature.corner_peaks(np_corner_response, min_distance=25) # (row, column)
        indeces = random.choices(range(peaks.shape[0]), k=self.batch_size)
    
        return [warp(image, camera_parameters, x, y, T.InterpolationMode.BILINEAR) for y, x in peaks[indeces]]