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

class CameraPreservingSampler(PatchSampler):
    def __init__(
            self,

            # Augmentation options
            color_jittering: bool,
            hflip: bool,
            scaling_range: tuple[float, float],

            batch_size: int,

            # Patch options
            blur: bool,
            h: int,
            w: int,
            #corner_sampling: bool,
        ):
        self.color_jittering = color_jittering
        self.hflip = hflip
        self.scaling_range = scaling_range

        self.batch_size = batch_size

        self.h = h
        self.w = w
        self.blur = blur
        #self.corner_sampling = corner_sampling
        #if self.corner_sampling:
        #    self.sample_patches = self.sample_corner
        #else:
        #    self.sample_patches = self.random_sampling

        self.color_jitter = T.Compose([
            T.ColorJitter(0.1, 0.1, 0.1, 0.),
        ])

        self.training = True

    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:

        # Sample points of interest and warp the image and depth centering them
        samples: list[tuple[Tensor, dict[str, Tensor]]] = self.sample_patches(image, camera_parameters)

        out_images: list[Tensor] = []
        out_camera_parameters: list[dict[str, Tensor]] = []
        out_depth_maps: list[Tensor] = []
        for img, cam_params in samples:
            
            out_img, out_cam_params = img, cam_params

            # Blur
            if self.blur:
                out_img = blur(out_img)
            
            if self.training:
                # Augmentations
                out_img = self.color_jitter(out_img)

                flip = random.random() > 0.5
                if self.hflip and flip:
                    out_img, out_camera_parameters = horizontal_flip_through_camera(out_img, out_camera_parameters)

                s = random.uniform(*self.scaling_range)
                out_img, out_camera_parameters = scale_through_depth(out_img, out_camera_parameters, s)

            # Center crop
            out_img, out_cam_params = center_crop_through_camera(img, cam_params, (self.h, self.w))

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

    #def random_sampling(
    #    self, image: Tensor, camera_parameters: dict[str, Tensor]
    #) -> list[tuple[Tensor, dict[str, Tensor]]]:
    #    
    #    # Randomly select point
    #    p1 = 0.2 # TODO: p1, p2 should depend on image size
    #    p2 = 0.8
    #    def sample_point() -> tuple[int, int]:
    #        x = random.randrange(int(p1 * image.shape[-1]), int(p2 * image.shape[-1]))
    #        y = random.randrange(int(0.4 * image.shape[-2]), int(p2 * image.shape[-2]))
    #        return x, y
    #
    #    return [warp(image, camera_parameters, *sample_point()) for _ in range(self.batch_size)]
    
    def sample_patches(
        self, image: Tensor, camera_parameters: dict[str, Tensor]
    ) -> list[tuple[Tensor, dict[str, Tensor]]]:

        # Sample peaks of interest
        peaks: Tensor = self.detect_points_of_interest(image)

        if self.training:
            indeces = random.choices(range(peaks.shape[0]), k=self.batch_size)
            peaks = peaks[indeces]

        return [self.get_patch(image, camera_parameters, float(x.item()), float(y.item())) for y, x in peaks]
    
    def get_patch(self, image: Tensor, camera_parameters: dict[str, Tensor], x: float, y: float) -> tuple[Tensor, dict[str, Tensor]]:
        # Warp and crop
        w_image, w_camera_parameters = warp(image, camera_parameters, x, y, T.InterpolationMode.BILINEAR)
        c_w_image, c_w_camera_parameters = center_crop_through_camera(w_image, w_camera_parameters, (self.h, self.w))
        c_w_image = c_w_image if not self.blur else blur(c_w_image)

        return c_w_image, c_w_camera_parameters
    
    def detect_points_of_interest(self, image: Tensor) -> Tensor:
        """Returns the coordinates of the points of interest in the image.
        
        The result is a tensor of shape N x 2 corresponding to rows and columns of the image.
        The result is on the cpu device.
        """
        # device: torch.device = image.device

        np_image = image.permute(1, 2, 0).cpu().numpy()
        np_corner_response = skimage.feature.corner_moravec(skimage.color.rgb2gray(np_image))
        np_corner_response = clean_corner_response(np_corner_response)

        # Sample peaks of interest
        peaks: np.ndarray = skimage.feature.corner_peaks(np_corner_response, min_distance=image.shape[-2] // 15) # (row, column)

        return torch.from_numpy(peaks)#.to(device)