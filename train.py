import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from training.models.eigen import CoarseFine
from training.losses.scale_invariant import ScaleInvariantLoss
from training.kitti_data.kitti_raw.kitti_dataset import KITTIRAWDataset
from training.kitti_data.augmenters import EigenAugmenter
from training.kitti_data.patch_samplers import EigenPatchSampler

def main():
    model = CoarseFine()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ScaleInvariantLoss()
    train_set = KITTIRAWDataset(
        EigenAugmenter(),
        EigenPatchSampler(16),
        '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti',
        'train',
        'png',
    )
    val_set = KITTIRAWDataset(
        EigenAugmenter(),
        EigenPatchSampler(16),
        '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti',
        'val',
        'png',
    )

    for epoch in range(10):
        model.to('cuda')
        model.train()
        for image, depth_map, valid_mask, overlap_mask in tqdm(train_set):
            image = image.to('cuda')
            depth_map = depth_map.to('cuda')
            valid_mask.to('cuda')
            optimizer.zero_grad()
            pred = model(image, only_coarse=True)
            pred = F.resize(pred, depth_map.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
            loss = criterion(pred, depth_map)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()