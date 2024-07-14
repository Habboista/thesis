from src.models.eigen import CoarseFine
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
kitti_path = '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti'
kitti_ext = 'png'
experiment_name = __name__.split('.')[-1]

# Training options
num_epochs = 10
batch_size = 64
lr = 1e-4
criterion = ScaleInvariantLoss()

# Model
model = CoarseFine()

# Data manipulation options
train_augmenter = EigenAugmenter()
train_patch_sampler = EigenPatchSampler(1)
val_augmenter = TestAugmenter()
val_patch_sampler = TestPatchSampler()