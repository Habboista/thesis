import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from ..info import Info
from ..models import Model
from .train import train
from ..eval import eval
from ..data.kitti_raw import KITTIRAWDataset

def train_cycle(
    model: Model,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_dataset: KITTIRAWDataset,
    val_dataset: KITTIRAWDataset,
    start_epoch: int,
    num_epochs: int,
    batch_size: int,
    training_info: Info,
    val_info: Info,
    checkpoints_dir: str
):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        training_info.new_epoch()
        val_info.new_epoch()

        # Training
        print("Training...")
        train(model, optimizer, criterion, train_loader, training_info)
        training_info.print_last_epoch_summary()

        # Validation
        print("Validation...")
        eval(model, val_dataset, val_info)
        val_info.print_last_epoch_summary()

        # Save checkpoint
        print("Saving checkpoint...")
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'model_{epoch}.pth'))
        training_info.save_last_epoch(f'training_info_{epoch}.json')
        val_info.save_last_epoch(f'val_info_{epoch}.json')