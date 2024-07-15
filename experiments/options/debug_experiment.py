from src.models import MyModel
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
kitti_path = '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti'
kitti_ext = 'png'
experiment_name = __name__.split('.')[-1]

# Training options
num_epochs = 10
batch_size = 1
lr = 1e-4
criterion = ScaleInvariantLoss()

# Model
model = MyModel()

# Data manipulation options
train_augmenter = MyAugmenter(
    color_jittering=True,
    hflip=True,
    scaling_range=(0.5, 1.),
)
train_patch_sampler = MyPatchSampler(
    batch_size=16,
    blur=True,
    half_h=80,
    half_w=80,
    corner_sampling=True,
)
val_augmenter = MyAugmenter(
    color_jittering=False,
    hflip=False,
    scaling_range=(1., 1.),
)
val_patch_sampler = MyPatchSampler(
    batch_size=1,
    blur=True,
    half_h=80,
    half_w=80,
    corner_sampling=True,
)