import os

import torch

from src.training import train_cycle
from src.data.kitti_raw import KITTIRAWDataset
from src.info import Info

########################################################
#                   EXPERIMENT OPTIONS                 #
from experiments.options.eigen import *
########################################################

def main():
    global \
        kitti_path,\
        kitti_ext,\
        experiment_name,\
        num_epochs,\
        batch_size,\
        lr,\
        criterion,\
        model,\
        train_augmenter,\
        train_patch_sampler,\
        val_augmenter,\
        val_patch_sampler
    
    # Directories
    experiment_dir = os.path.join('experiments', experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    info_dir = os.path.join(experiment_dir, 'info')

    # Setup
    assert torch.cuda.is_available(), 'CUDA is not available.'
    assert os.path.isdir(kitti_path), f'Directory {kitti_path} does not exist.'
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)

    # Load checkpoint
    print("Looking for a checkpoint...")
    checkpoints = os.listdir(checkpoints_dir)
    if len(checkpoints) > 0:
        checkpoints.sort()
        checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
        print(f"Checkpoint {checkpoints[-1]} found, loading it.\n" \
            f"Resuming from epoch: {len(checkpoints)}\n")
        model.load_state_dict(torch.load(checkpoint_path))
        num_epochs -= len(checkpoints)
    else:
        print("No checkpoint found, instantiating new model.\n")

    model = model.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Data
    train_set = KITTIRAWDataset(
        train_augmenter,
        train_patch_sampler,
        kitti_path,
        'train',
        kitti_ext,
    )
    
    training_info = Info(info_dir, periodic_plot=True, period=1024 // batch_size)

    val_set = KITTIRAWDataset(
        val_augmenter,
        val_patch_sampler,
        kitti_path,
        'val',
        kitti_ext,
    )

    val_info = Info(info_dir)

    # Train cycle
    train_cycle(
        model,
        optimizer,
        criterion,
        train_set,
        val_set,
        num_epochs,
        batch_size,
        training_info,
        val_info,
        checkpoints_dir,
    )

if __name__ == "__main__":
    main()