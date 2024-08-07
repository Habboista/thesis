import os

import torch

from src.eval import eval_pipeline
from src.data.kitti_raw import KITTIRAWDataset
from src.data.augmenters import NoAugmenter
from src.data.patch_samplers import NoPatchSampler
from src.info import Info

########################################################
#                   EXPERIMENT OPTIONS                 #
from experiments.options.debug_experiment import *
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
    print("Make sure to run the corresponding experiment first")
    assert os.path.isdir(experiment_dir), f'Directory {experiment_dir} does not exist.'
    assert os.path.isdir(checkpoints_dir), f'Directory {checkpoints_dir} does not exist.'
    assert os.path.isdir(info_dir), f'Directory {info_dir} does not exist.'

    # Load checkpoint
    print("Looking for a checkpoint...")
    checkpoints = os.listdir(checkpoints_dir)
    if len(checkpoints) > 0:
        checkpoints.sort()
        checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
        print(f"Checkpoint {checkpoints[-1]} found, loading it.\n")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint found.\n")
        exit()

    model = model.to('cuda')

    eval_set = KITTIRAWDataset(
        NoAugmenter(),
        NoPatchSampler(),
        kitti_path,
        'val',
        kitti_ext,
    )

    eval_info = Info(info_dir)

    eval_pipeline(
        model,
        eval_set,
        train_patch_sampler,
        20,
        1,
        0.1,
        eval_info,
    )

if __name__ == "__main__":
    main()