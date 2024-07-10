import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F

from .abstract_patch_sampler import PatchSampler
from ..point_cloud import PointCloud


class NoPatchSampler(PatchSampler):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_shape: tuple[int, int] = (image.shape[-2], image.shape[-1])
        depth_map: Tensor = torch.from_numpy(point_cloud.to_depth_map(image_shape))
        valid_mask: Tensor = (depth_map > 0)

        return image.unsqueeze(0), depth_map.unsqueeze(0), valid_mask.unsqueeze(0), torch.zeros(1, 1, *image_shape)


class EigenPatchSampler(PatchSampler):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.crop_size = (172, 576)

    def __call__(self, image: Tensor, point_cloud: PointCloud) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s = random.uniform(0.5, 0.75)
        flip = random.random() > 0.5
        shape: tuple[int, int] = (image.shape[-2], image.shape[-1])
        image = F.resize(image, (int(shape[0] * s), int(shape[1] * s)), interpolation=F.InterpolationMode.BILINEAR)
        image = F.hflip(image) if flip else image

        depth_map: Tensor = torch.from_numpy(point_cloud.to_depth_map(shape)).unsqueeze(0) / s
        depth_map = F.hflip(depth_map) if flip else depth_map
        depth_map = F.resize(depth_map, (int(shape[0] * s), int(shape[1] * s)), interpolation=F.InterpolationMode.NEAREST).squeeze(0)

        valid_mask: Tensor = (depth_map > 0)

        batched_image: list[Tensor] = []
        batched_depth_map: list[Tensor] = []
        batched_valid_mask: list[Tensor] = []
        for _ in range(self.batch_size):
            i = random.randint(0, int(s*shape[0]) - self.crop_size[0] - 1)
            j = random.randint(0, int(s*shape[1]) - self.crop_size[1] - 1)

            
            batched_image.append(image[:, i:i+self.crop_size[0], j:j+self.crop_size[1]])
            batched_depth_map.append(depth_map[i:i+self.crop_size[0], j:j+self.crop_size[1]])
            batched_valid_mask.append(valid_mask[i:i+self.crop_size[0], j:j+self.crop_size[1]])

        image = torch.stack(batched_image)
        depth_map = torch.stack(batched_depth_map)
        valid_mask = torch.stack(batched_valid_mask)

        return image, depth_map, valid_mask, torch.zeros(1, 1, *self.crop_size)
