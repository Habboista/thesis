from abc import ABC, abstractmethod
from typing import Iterator
import os

from torch import Tensor
import torch.utils.data as torch_data

from ..augmenters import Augmenter
from ..patch_samplers import PatchSampler
from ..point_cloud import PointCloud

def readlines(filename: str) -> list[str]:
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

class EigenSplitDataset(ABC, torch_data.Dataset):
    """Base class for KITTI dataset. It provides a common interface to the Eigen split of KITTI.
    """
    def __init__(
            self,
            augmenter: Augmenter,
            patch_sampler: PatchSampler,
            data_path: str,
            mode: str,
            img_ext: str,
    ):
        self.augmenter: Augmenter = augmenter
        self.patch_sampler: PatchSampler = patch_sampler
        self.data_path: str = data_path

        match mode:
            case "train":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", "train_files.txt")
            case "val":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", "val_files.txt")
            case "test":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", "test_files.txt")
            case _:
                raise ValueError(f"Unknown mode {mode}")
            
        self.filenames: list[str] = readlines(filenames_path)
        self.img_ext: str = img_ext

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image, point_cloud = self._load(index)
        image, point_cloud = self.augmenter(image, point_cloud)
        image_patches, depth_patches = self.patch_sampler(image, point_cloud)
        
        return image_patches, depth_patches

    @abstractmethod
    def _load(self, index: int) -> tuple[Tensor, PointCloud]:
        """Load an image and corresponding point cloud from file"""
        ...