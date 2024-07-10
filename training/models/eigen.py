import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class Coarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(4096, 27*142)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).reshape(-1, 1, 27, 142)

class Fine(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fine_0_1 = nn.Conv2d(3, 63, 9, stride=2)
        self.fine_2_3 = nn.Conv2d(64, 64, 5, padding='same', padding_mode='zero')
        self.fine_3_4 = nn.Conv2d(64, 1, 5, padding='same', padding_mode='zero')
    
    def forward(self, x: Tensor, xx: Tensor) -> Tensor:
        fine_1 = self.pool(self.fine_0_1(x))
        fine_2 = self.act(torch.cat((x, xx), dim=-3))
        fine_3 = self.act(self.fine_2_3(fine_2))
        fine_4 = self.fine_3_4(fine_3)
        return fine_4

class CoarseFine(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = Coarse()
        self.fine = Fine()

    def forward(self, x: Tensor, only_coarse: bool = False) -> Tensor:
        if only_coarse:
            return self.coarse(x)
        return self.fine(x, self.coarse(x))