import os
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms.functional as F
from tqdm import tqdm

from training.models.eigen import CoarseFine
from training.eval import eval
from training.losses.scale_invariant import ScaleInvariantLoss
from training.kitti_data.kitti_raw.kitti_dataset import KITTIRAWDataset
from training.kitti_data.augmenters import NoAugmenter
from training.kitti_data.augmenters import EigenAugmenter
from training.kitti_data.patch_samplers import EigenPatchSampler
from training.kitti_data.patch_samplers import NoPatchSampler

def train(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: data.DataLoader
    ) -> dict[str, list[float]]:

    training_info: dict[str, list[float]] = {'train_loss': [], 'train_reg': []}
    model.train()
    k = 0
    for image, depth_map in tqdm(train_loader):
        if k > 10:
            break
        k += 1

        optimizer.zero_grad()

        # convert to cuda
        image = image.to('cuda')
        depth_map = depth_map.to('cuda')

        # inputs have two leading batching dimensions
        image = image.reshape(-1, *image.shape[2:])
        depth_map = depth_map.reshape(-1, *depth_map.shape[2:])

        # predict
        pred = model(image)

        # regularization
        reg = torch.tensor(0., device='cuda')
        for n, p in enumerate(model.parameters()):
            reg += torch.sum(torch.abs(p)) / p.numel()
        reg /= n

        # compute loss
        mask = depth_map > 0
        train_loss = criterion(pred[mask], torch.log(depth_map[mask]))
        loss: Tensor = train_loss + 0.01 * reg
        
        # backpropagation
        loss.backward()
        optimizer.step()

        training_info['train_loss'].append(train_loss.detach().cpu().item())
        training_info['train_reg'].append(reg.detach().cpu().item())
        print(f"Train loss: {train_loss:.4f}, reg: {reg:.4f}")
    
    return training_info

def main():
    assert torch.cuda.is_available(), 'CUDA is not available.'

    model = CoarseFine().to('cuda')
    training_info = {'train_loss': [], 'train_reg': []}
    # Load checkpoint
    ...

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ScaleInvariantLoss()

    train_set = KITTIRAWDataset(
        EigenAugmenter(),
        EigenPatchSampler(1),
        '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti',
        'train',
        'png',
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    val_set = KITTIRAWDataset(
        NoAugmenter(),
        NoPatchSampler(),
        '/media/antonio/523f31c5-dc82-4dce-8457-65b5dd1f19e4/kitti',
        'val',
        'png',
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    num_epochs = 3
    best_total_loss = float('inf')
    checkpoints_path = os.path.join('experiments', 'eigen', 'checkpoints') 
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        training_info = train(model, optimizer, criterion, train_loader)

        # Average training loss
        total_loss = sum(training_info['train_loss']) / len(training_info['train_loss'])
        print("Train loss:", total_loss)

        # Validation
        val_info = eval(model, val_loader)
        for k, v in val_info.items():
            print(f"{k}: {v}")

        if total_loss < best_total_loss:
            # Save checkpoint
            best_total_loss = total_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f'best_model_{epoch}.pth'))
        
if __name__ == "__main__":
    main()