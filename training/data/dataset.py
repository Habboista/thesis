from abc import ABC, abstractmethod
from typing import Iterator

from torch import Tensor
import torch.utils.data as torch_data

from . import Augmenter
from . import PatchSampler

class Dataset(ABC, torch_data.IterableDataset):
    def __init__(
            self,
            augmenter: Augmenter,
            patch_sampler: PatchSampler,
            data_path: str,
            filenames: list[str]
    ):
        self.augmenter: Augmenter = augmenter
        self.patch_sampler: PatchSampler = patch_sampler
        self.index: int = -1
        self.data_path: str = data_path
        self.filenames: list[str] = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __iter__(self) -> Iterator:
        if self.index < 0:
            self.index = 0
            return self
        raise IndexError("Dataset is already being iterated")
    
    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.index >= len(self):
            raise StopIteration

        image, point_cloud = self._load(self.index)
        image, point_cloud = self.augmenter(image, point_cloud)
        image_patches, depth_patches, valid_masks, overlap_masks = self.patch_sampler(image, point_cloud)
        
        self.index += 1
        return image_patches, depth_patches, valid_masks, overlap_masks

    @abstractmethod
    def _load(self, index: int) -> tuple[Tensor, Tensor]:
        ...