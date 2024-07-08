from torch import Tensor
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class Coarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(4096, 27*142)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Fine(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class CoarseFine(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = Coarse()
        self.fine = Fine()

    def forward(self, x: Tensor, only_coarse: bool = False) -> Tensor:
        if only_coarse:
            return self.coarse(x)
        return #self.fine(self.coarse(x))