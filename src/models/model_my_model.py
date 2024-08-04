import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ._model_abstract import Model
from .model_eigen import CoarseFine
from ..data.transforms import scale_through_depth, center_crop_through_camera, cloud2depth, depth2cloud, warp
from ..data.patch_samplers import blur, get_blur_weight_mask, clean_corner_response

class MyModel(Model):
    def __init__(self, blur: bool, num_epochs: int):
        super().__init__()
        self.base_model = CoarseFine(coarse_size=(32, 32))
        self.blur: bool = blur
        self.weight: Tensor = torch.tensor(1.) if not self.blur else get_blur_weight_mask((150, 150))
        self.num_epochs = num_epochs
    
    def _copy_model(self) -> Model:
        model: Model = CoarseFine(coarse_size=(32, 32)).to(self.device)
        for param_dst, param_src in zip(self.model.parameters(), self.base_model.parameters()):
            param_dst.data.copy_(param_src.data)
        return model
    
    def _predict(self, image: Tensor, camera_parameters: dict[str, Tensor], model: Model) -> list[Tensor]:
        # Detect points of interest
        print("Detecting points of interest...")
        points_of_interest: Tensor = torch.empty(100, 2)
        print(f"Fround {points_of_interest.shape[0]} points of interest")

        h: int = 150
        w: int = 150

        depth_list: list[Tensor] = []
        for x, y in points_of_interest:
            print(f"\nWorking on point ({x}, {y})")

            # Warp and crop
            print("Warp and crop...")
            w_image, w_camera_parameters = warp(image, camera_parameters, x, y)
            c_w_image, c_w_camera_parameters = center_crop_through_camera(w_image, w_camera_parameters, (h, w))
            c_w_image = c_w_image if not self.blur else blur(c_w_image)
            
            # Predict
            print("Predict...")
            depth: Tensor = model(c_w_image[None], c_w_camera_parameters)

            # Unwarp
            print("Unwarp...")
            cloud = depth2cloud(depth, c_w_camera_parameters)
            depth = cloud2depth(cloud, camera_parameters)

            depth_list.append(depth)

        return depth_list
    
    def _compute_loss(self, preds: list[Tensor], first_preds: list[Tensor]) -> Tensor:
        loss: Tensor
        reg: Tensor
        masks = [depth > 0 for depth in preds]
        for i in range(len(preds)):
            for j in range(len(preds)):
                if i == j:
                    reg = reg + torch.mean(self.weight * (preds[i][masks[i]] - first_preds[i][masks[i]])**2)
                else:
                    loss = loss + torch.mean((preds[i][masks[i]] - preds[j][masks[j]])**2)
        loss = loss + reg
        return loss
        
    def _forward(self, x: Tensor, camera_parameters: dict[str, Tensor]) -> Tensor:
        if self.training:
            return self.base_model(x, camera_parameters)

        # Work with a copy of the base model
        model: Model = self._copy_model()

        # Unbatch (Expected batch of size 1)
        print("Unbatching...")
        assert len(x.shape) == 4, "Expected input of shape 1 x 3 x H x W"
        assert x.shape[0] == 1, "Expected input of shape 1 x 3 x H x W"
        image = x[0]
        for k in camera_parameters:
            camera_parameters[k] = camera_parameters[k][0]

        print("Computing the first predictions...")
        with torch.no_grad():
            first_preds: list[Tensor] = self._predict(image, camera_parameters, self.base_model)

        # Fine tune the network
        print("Fine-tuining before blending...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            model.zero_grad()
            preds = self._predict(image, camera_parameters, model)
            loss = self._compute_loss(preds, first_preds)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param.data -= 0.1 * param.grad

        # Blend
        print("\nComputing the final predictions...")
        with torch.no_grad():
            preds = self._predict(image, camera_parameters, model)
        
        print("Blending...")
        stacked_preds: Tensor = torch.stack(preds)
        stacked_preds[stacked_preds <= 0] = torch.nan

        return stacked_preds.nanmean(0)