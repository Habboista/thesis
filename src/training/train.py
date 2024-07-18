import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from ..info import Info

def train(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: data.DataLoader,
        training_info: Info,
    ) -> None:

    model.train()
    for image, depth_map, camera_parameters in tqdm(train_loader):
        optimizer.zero_grad()

        # convert to cuda
        image = image.to('cuda')
        depth_map = depth_map.to('cuda')

        # inputs have two leading batch dimensions
        image = image.reshape(-1, *image.shape[-3:])
        depth_map = depth_map.reshape(-1, *depth_map.shape[-3:])
        camera_parameters = {
            k: v.reshape(-1, *v.shape[-2:]).to('cuda')
            for k, v in camera_parameters.items()}

        # predict
        pred = model(image, camera_parameters)

        # regularization
        reg: Tensor = torch.tensor(0., device='cuda')
        for n, p in enumerate(model.parameters()):
            reg += torch.sum(torch.abs(p)) / p.numel()
        reg /= n

        # compute loss
        train_loss: Tensor = criterion(pred, depth_map)
        loss: Tensor = train_loss + 0.01 * reg
        
        # backpropagation
        loss.backward()
        optimizer.step()

        training_info.log_info({
            'train_loss': train_loss.detach().cpu().numpy().item(),
            'train_reg': reg.detach().cpu().numpy().item(),
        })