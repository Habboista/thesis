from typing import Callable

import skimage
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ..models._model_abstract import Model
from ..models.model_eigen import CoarseFine
from ..data.transforms import scale_through_depth, center_crop_through_camera, cloud2depth, depth2cloud, warp
from ..data.patch_samplers import blur as apply_blur
from ..data.patch_samplers import get_blur_weight_mask, clean_corner_response

def infer(image: Tensor, camera_parameters: dict[str, Tensor], model: Model, patch_sampler: Callable) -> Tensor:
    # Work with a copy of the model
    device: torch.device = image.device
    model = copy_model(model, device)

    # Get patches to be used for inference
    patch_list, camera_params_list = get_patches(image, camera_parameters, 32, 32, False)

    # Compute first partial predictions
    model.eval()
    with torch.no_grad():
        pred_list = batch_predict(patch_list, camera_params_list, model, camera_parameters)
    
    # Fine tune the network
    fine_tune(model, camera_parameters, patch_list, camera_params_list, pred_list, 2)

    # Compute final partial predictions
    model.eval()
    with torch.no_grad():
        pred_list = batch_predict(patch_list, camera_params_list, model, camera_parameters)

    # Blend partial predictions
    return blend(pred_list)

def blend(preds: list[Tensor]) -> Tensor:
    print("Blending...")
    stacked_preds: Tensor = torch.stack(preds)
    stacked_preds[stacked_preds <= 0] = torch.nan

    result = stacked_preds.nanmean(0).unsqueeze(0)
    result[result.isnan() | (result <= 0)] = 1e-3

    return result

def fine_tune(model: Model, camera_parameters: dict[str, Tensor], patch_list: list[Tensor], params_list: list[dict[str, Tensor]], first_preds: list[Tensor], num_epochs: int) -> None:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        preds = batch_predict(patch_list, params_list, model, camera_parameters) # linear-scale
        loss = compute_loss(preds, first_preds)
        loss.backward()
        optimizer.step()

def compute_loss(preds: list[Tensor], first_preds: list[Tensor]) -> Tensor:
    loss: Tensor = torch.tensor(0., requires_grad=True, device='cuda')
    reg: Tensor = torch.tensor(0., requires_grad=True, device='cuda')
    masks = [depth > 0 for depth in preds]
    for i in range(len(preds)):
        for j in range(len(preds)):
            if i == j:
                # TODO use self.weight
                reg = reg + torch.mean((preds[i][masks[i]] - first_preds[i][masks[i]])**2)
            else:
                m = masks[i] & masks[j]
                loss = loss + torch.mean((preds[i][m] - preds[j][m])**2)
    loss = loss + reg
    return loss

def batch_predict(patch_list: list[Tensor], camera_params_list: list[dict[str, Tensor]], model: Model, starting_camera_parameters: dict[str, Tensor]) -> list[Tensor]:
    pred_list: list[Tensor] = []

    for patch, camera_params in zip(patch_list, camera_params_list):
        # Predict
        print("Predict...")
        depth: Tensor
        if model.training: # Model behaviour change based on training mode, I want linear scale output
            depth = torch.exp(model(patch[None], camera_params)) # log to linear
        else:
            depth = model(patch[None], camera_params) # already in linear

        # Unwarp
        print("Unwarp...")
        depth = depth[0]
        cloud = depth2cloud(depth, camera_params)
        pred = cloud2depth(cloud, starting_camera_parameters)
    
        pred_list.append(pred)
    
    return pred_list

def copy_model(src_model: Model, device: torch.device) -> Model:
    model: Model

    if isinstance(src_model, CoarseFine):
        model = CoarseFine(coarse_size=(32, 32)).to(device)
        for param_dst, param_src in zip(model.parameters(), src_model.parameters()):
            param_dst.data.copy_(param_src.data)
        return model
    else:
        raise NotImplementedError

def get_patches(
    image: Tensor, camera_parameters: dict[str, Tensor], h: int, w: int, blur: bool
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    # Detect points of interest
    print("Detecting points of interest...")
    points_of_interest: Tensor = detect_points_of_interest(image)
    print(f"Found {points_of_interest.shape[0]} points of interest")

    patch_list: list[Tensor] = []
    camera_params_list: list[dict[str, Tensor]] = []
    for y, x in points_of_interest:
        print(f"\nWorking on point (x={x}, y={y})")

        # Warp and crop
        print("Warp and crop...")
        w_image, w_camera_parameters = warp(image, camera_parameters, x, y, T.InterpolationMode.BILINEAR)
        c_w_image, c_w_camera_parameters = center_crop_through_camera(w_image, w_camera_parameters, (h, w))
        c_w_image = c_w_image if not blur else apply_blur(c_w_image)

        patch_list.append(c_w_image)
        camera_params_list.append(c_w_camera_parameters)

    return patch_list, camera_params_list

def detect_points_of_interest(image: Tensor) -> Tensor:
    device: torch.device = image.device

    np_image = image.permute(1, 2, 0).cpu().numpy()
    np_corner_response = skimage.feature.corner_moravec(skimage.color.rgb2gray(np_image))

    np_corner_response = clean_corner_response(np_corner_response)

    # Sample peaks of interest
    peaks = skimage.feature.corner_peaks(np_corner_response, min_distance=15)[:20, :] # (row, column)

    return torch.from_numpy(peaks).to(device)