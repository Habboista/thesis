from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

class AbstractModel(Module, ABC):
    """Base class for all models. It ensures input and output format."""
    def __init__(self):
        super().__init__()

    def to(self, device):
        super().to(device)

    def train(self):
        super().train()
    
    def eval(self):
        super().eval()

    @abstractmethod
    def _forward(self, image: Tensor) -> Tensor:
        ...

    def _check_inputs(self, image: Tensor):
        if type(image) != Tensor:
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
        
        if len(image.shape) != 4:
            raise ValueError(f"image must be a 4D tensor of shape B x 3 x H x W, got {image.shape}")
        if image.shape[1] != 3:
            raise ValueError(f"image must be a 4D tensor of shape B x 3 x H x W, got {image.shape}")
        
        self.output_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
    
    def _check_outputs(self, result: Tensor):
        if len(result.shape) != 4:
            raise ValueError(f"result must be a 4D tensor of shape {self.output_shape}, got {result.shape}")
        if result.shape[1] != 1:
            raise ValueError(f"result must be a 4D tensor of shape {self.output_shape}, got {result.shape}")
        if result.shape != self.output_shape:
            raise ValueError(f"result must be a 4D tensor of shape {self.output_shape}, got {result.shape}")
        
    def forward(self, image: Tensor) -> Tensor:
        self._check_inputs(image)
        result = self._forward(image)
        self._check_outputs(result)
        return result
    
    def __call__(self, image: Tensor) -> Tensor:
        return self.forward(image)