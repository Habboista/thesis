import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._model_abstract import Model
from .model_eigen import CoarseFine
from ..data.transforms import scale_through_depth, get_rotation_matrix, get_start_and_end_points

class MyModel(Model):
    def __init__(self, blur: bool):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(32, 32))
        self.blur: bool = blur
    
    def _copy_model(self) -> Model:
        model: Model = CoarseFine(coarse_size=(32, 32)).to(self.device)
        for param_dst, param_src in zip(self.model.parameters(), self.base_model.parameters()):
            param_dst.data.copy_(param_src.data)
        return model
    
    def _predict(self, image: Tensor, camera_parameters: dict[str, Tensor], model: Model) -> tuple[Tensor, Tensor]:
        # From coarser scale to finer TODO

        # Detect points of interest
        points_of_interest: Tensor = torch.empty(100, 2)
        x: int
        y: int

        # Warp
        px: int = int(camera_parameters['K'][0, 2])
        py: int = int(camera_parameters['K'][1, 2])
        dh: int = 10
        dw: int = 10
        R = get_rotation_matrix(x, y, camera_parameters)
        start_points, end_points = get_start_and_end_points(R, camera_parameters)
        warped_image = \
            F.perspective(image, start_points, end_points, interpolation=T.InterpolationMode.BILINEAR) \
            [..., py - dh:py + dh, px - dw:px + dw]
            
        # Predict
        pred: Tensor = model(warped_image[None], camera_parameters)

        ##########################
        #        Warp back       #
        ##########################

        # Identify pixels on which the prediction was done in the original image
        center_mask = torch.zeros(image.shape[-2:], dtype=torch.bool)
        center_mask[py - dh:py + dh, px - dw:px + dw] = True
        warped_mask: Tensor = \
            F.perspective(center_mask, end_points, start_points, interpolation=T.InterpolationMode.NEAREST)
        
        # Sample depth values from the predicted depth map
        ...

        preds: Tensor
        masks: Tensor
        return preds, masks
        
    def _forward(self, x: Tensor, camera_parameters: dict[str, Tensor]) -> Tensor:
        if self.training:
            return self.base_model(x, camera_parameters)

        # Work with a copy of the base model
        model: Model = self._copy_model()

        # Unbatch (Expected batch of size 1)
        assert len(x.shape) == 4, "Expected input of shape 1 x 3 x H x W"
        assert x.shape[0] == 1, "Expected input of shape 1 x 3 x H x W"
        image = x[0]
        for k in camera_parameters:
            camera_parameters[k] = camera_parameters[k][0]
                
        ##########################
        #          Blend         #
        ##########################
        
        weight: Tensor
        masks: list[Tensor]
        preds: list[Tensor]
        first_preds: list[Tensor]

        # Fine tune the network

        # Loss
        loss: Tensor
        reg: Tensor
        for i in range(len(masks)):
            for j in range(len(masks)):
                if i == j:
                    reg = reg + torch.mean(weight * (pred[i][masks[i]] - first_preds[i][masks[i]])**2)
                else:
                    loss = loss + torch.mean((pred[i][masks[i]] - preds[j][masks[j]])**2)
        loss = loss + reg

        # Blend
        stacked_masks: Tensor = torch.stack(masks)
        stacked_preds: Tensor = torch.stack(preds)
        stacked_preds[~stacked_masks] = torch.nan

        return stacked_preds.nanmean(0)