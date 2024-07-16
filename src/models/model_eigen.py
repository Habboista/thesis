import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as torch_models

from ._model_abstract import Model

class Coarse(nn.Module):
    def __init__(self, out_size: tuple[int, int]):
        super().__init__()
        self.out_size = out_size
        self.add_module(
            'backbone',
            torch_models.alexnet(weights=torch_models.AlexNet_Weights.DEFAULT)
        )
        del self.backbone.classifier[-1]
        self.add_module('head', nn.Linear(4096, out_size[0] * out_size[1]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        x = x.reshape(-1, 1, self.out_size[0], self.out_size[1])
        return x

class Fine(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fine_0_1 = nn.Conv2d(3, 63, 9, stride=2)
        self.fine_2_3 = nn.Conv2d(64, 64, 5, padding='same', padding_mode='zeros')
        self.fine_3_4 = nn.Conv2d(64, 1, 5, padding='same', padding_mode='zeros')
    
    def forward(self, x: Tensor, coarse_map: Tensor) -> Tensor:
        fine_1 = self.pool(self.fine_0_1(x))
        fine_1 = nn.functional.interpolate(fine_1, size=coarse_map.shape[-2:], mode='nearest')
        fine_2 = self.act(torch.cat((fine_1, coarse_map), dim=-3))
        fine_3 = self.act(self.fine_2_3(fine_2))
        fine_4 = self.fine_3_4(fine_3)
        return fine_4

class CoarseFine(Model):
    def __init__(self, coarse_size: tuple[int, int]=(27, 142)):
        super().__init__()
        self.coarse = Coarse(coarse_size)
        self.fine = Fine()

    def _forward(self, x: Tensor) -> Tensor:
        x = 2* (x - 0.5)
        coarse_map = self.coarse(x)
        fine_map = self.fine(x, coarse_map)
        fine_map = nn.functional.interpolate(fine_map, size=x.shape[-2:], mode='nearest')
        return torch.exp(fine_map) if not self.training else fine_map