from src.models import Model
from src.losses import Loss
from src.data.augmenters import Augmenter
from src.data.patch_samplers import PatchSampler

from src.models.model_eigen import CoarseFine
from src.losses import ScaleInvariantLoss

from src.data.augmenters import NoAugmenter
from src.data.augmenters import TestAugmenter
from src.data.augmenters import EigenAugmenter

from src.data.patch_samplers import EigenPatchSampler
from src.data.patch_samplers import TestPatchSampler
from src.data.patch_samplers import NoPatchSampler

__all__ = [
    'kitti_path',
    'kitti_ext',
    'experiment_name',
    'num_epochs',
    'batch_size',
    'lr',
    'criterion',
    'model',
    'train_augmenter',
    'train_patch_sampler',
    'val_augmenter',
    'val_patch_sampler',
]

# Setup options
kitti_path: str = '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti'
kitti_ext: str = 'png'
experiment_name: str = __name__.split('.')[-1]

# Training options
num_epochs: int = 10
batch_size: int = 64
lr: float = 1e-4
criterion: Loss = ScaleInvariantLoss()

# Model
model: Model = CoarseFine((27, 142))

# Data manipulation options
train_augmenter: Augmenter = EigenAugmenter()
train_patch_sampler: PatchSampler = EigenPatchSampler(1)
val_augmenter: Augmenter = TestAugmenter()
val_patch_sampler: PatchSampler = TestPatchSampler()