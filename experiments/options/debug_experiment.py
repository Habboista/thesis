from src.models import Model
from src.losses import Loss
from src.data.augmenters import Augmenter
from src.data.patch_samplers import PatchSampler

from src.models import CoarseFine
from src.losses import ScaleInvariantLoss

from src.data.augmenters import MyAugmenter
from src.data.patch_samplers import MyPatchSampler

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
batch_size: int = 1
lr: float = 1e-4
criterion: Loss = ScaleInvariantLoss()

# Model
model: Model = CoarseFine((32, 32))

# Data manipulation options
train_augmenter: Augmenter = MyAugmenter(
    color_jittering=True,
    hflip=True,
    scaling_range=(0.5, 1.),
)
train_patch_sampler: PatchSampler = MyPatchSampler(
    batch_size=16,
    blur=True,
    half_h=80,
    half_w=80,
    corner_sampling=True,
)
val_augmenter: Augmenter = MyAugmenter(
    color_jittering=False,
    hflip=False,
    scaling_range=(1., 1.),
)
val_patch_sampler: PatchSampler = MyPatchSampler(
    batch_size=1,
    blur=True,
    half_h=80,
    half_w=80,
    corner_sampling=True,
)